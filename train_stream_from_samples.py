"""
从 MaxCompute 样本表进行真正流式训练的独立入口脚本

使用方法示例：
    python train_stream_from_samples.py --config config/samples.conf --cuda 0

与 train_from_samples.py 的区别：
- 只负责 streaming 训练（不做全量预加载版本），逻辑更简洁
- 单独文件，避免影响原有 non-streaming 训练逻辑
- 日志命名严格遵守：train_YYYYMMDD_HHMMSS.log，每次运行新建文件
"""

import os
import math
import time
import argparse
import configparser
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.model import PatchSTG
from lib.utils import log_string, _compute_loss, loadDataFromSamples
from lib.patchstg_sample_loader import PatchSTGSampleLoader, build_locations_from_pairs


class StreamSampleSolver(object):
    """真正流式的 PatchSTG 训练 Solver（独立入口版本）。"""

    def __init__(self, config: dict):
        # 展开配置
        self.__dict__.update(**config)

        # 运行时间戳：YYYYMMDD_HHMMSS
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 日志命名规则：train_运行时间.log，且永远新建不覆盖
        self.log_file = self._build_log_path()
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        global log
        # 使用 "a" 也不会覆盖旧文件，但我们会通过唯一文件名保证不会复用
        log = open(self.log_file, "a")
        log_string(log, "======================STREAM CONFIG======================")
        for k, v in config.items():
            log_string(log, f"{k}: {v}")
        log_string(log, "========================================================")
        log_string(log, f"日志文件: {self.log_file}")

        # 初始化 loader（不做全量预加载）
        self.loader = PatchSTGSampleLoader(
            odps_project=self.odps_project,
            odps_table=self.odps_table,
            ds_partition=self.ds_partition,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            use_6dim_time_feat=getattr(self, "use_6dim_time_feat", False),
        )

        # 初始化空间结构：少量样本 + KDTree
        self._init_spatial_structure()

        # 设备
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        log_string(log, f"使用设备: {self.device}")

        # 模型 & 优化器
        self._build_model()

        self.best_epoch = 0
        # 初始化归一化统计
        self.mean = float(self.init_mean)
        self.std = float(self.init_std if self.init_std != 0 else 1.0)

    # -------------------- 日志相关 --------------------
    def _build_log_path(self) -> str:
        """生成日志文件路径：train_YYYYMMDD_HHMMSS.log

        如果配置文件中给了 log_dir，则放在该目录下，否则默认放在 ./log 目录。
        """
        # 支持从 config 传一个 log_dir 或 log_prefix，避免和原 train_from_samples 冲突
        log_dir = getattr(self, "log_dir", None)
        if not log_dir:
            # 如果原来配置的是一个文件名，也尽量复用目录部分
            base = getattr(self, "log_file", "log/train.log")
            base_dir = os.path.dirname(base) or "log"
            log_dir = base_dir
        filename = f"train_{self.run_timestamp}.log"
        return os.path.join(log_dir, filename)

    # -------------------- 空间结构初始化 --------------------
    def _init_spatial_structure(self):
        """读取一小部分样本，构建 locations + KDTree 索引 + 初始 mean/std。"""
        log_string(log, "==================INIT SPATIAL STRUCTURE==================")

        init_limit = getattr(self, "init_sample_limit", 1024)
        self.loader.load_from_odps(limit=init_limit)
        if not self.loader._samples:
            raise RuntimeError("初始化失败：未从样本表中读取到任何样本")

        data_dict = self.loader.prepare_training_data(normalize=False)
        metadata = data_dict["metadata"]
        self.sample_metadata = metadata

        unique_pairs = metadata["node_pairs"]

        # 路口位置信息
        if hasattr(self, "inter_location_dict") and self.inter_location_dict:
            self.loader.load_inter_locations(self.inter_location_dict)
        elif hasattr(self, "inter_location_table"):
            self.loader.load_inter_locations_from_odps(self.inter_location_table)
        else:
            log_string(log, "警告: 未提供路口位置信息,将无法进行空间划分")

        if self.loader.inter_id_locations:
            locations = build_locations_from_pairs(unique_pairs, self.loader.inter_id_locations)
            log_string(log, f"构建 locations 矩阵: shape={locations.shape}")
        else:
            log_string(log, "警告: 使用随机位置进行空间划分 (仅调试)")
            num_pairs = len(unique_pairs)
            locations = np.random.rand(2, num_pairs).astype(np.float32)

        # 调用 loadDataFromSamples 仅用于构建 KDTree 索引结构
        adjpath = os.path.join(self.model_file.replace(".pth", "_adj.npy"))
        (
            trainX,
            trainY,
            trainXTE,
            trainYTE,
            valX,
            valY,
            valXTE,
            valYTE,
            testX,
            testY,
            testXTE,
            testYTE,
            mean,
            std,
            self.ori_parts_idx,
            self.reo_parts_idx,
            self.reo_all_idx,
        ) = loadDataFromSamples(
            data_dict,
            locations,
            adjpath,
            self.recurtimes,
            self.spa_patchsize,
            log,
        )

        self.node_num = trainX.shape[2]
        log_string(log, f"节点数量 (max_nodes): {self.node_num}")

        # 保存初始化统计量
        self.init_mean = float(mean)
        self.init_std = float(std if std != 0 else 1.0)
        log_string(log, f"初始 Mean: {self.init_mean}, Std: {self.init_std}")
        log_string(log, "=======================================================")

    # -------------------- 模型构建 --------------------
    def _build_model(self):
        """构建 PatchSTG 模型及优化器。"""
        log_string(log, "======================BUILD MODEL======================")
        try:
            self.model = PatchSTG(
                self.output_len,
                self.tem_patchsize,
                self.tem_patchnum,
                self.node_num,
                self.spa_patchsize,
                self.spa_patchnum,
                getattr(self, "tod", 24),
                getattr(self, "dow", 7),
                self.layers,
                self.factors,
                self.input_dims,
                self.node_dims,
                getattr(self, "tod_dims", 8),
                getattr(self, "dow_dims", 8),
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx,
            ).to(self.device)
            log_string(log, f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        except Exception as e:
            log_string(log, f"模型构建失败: {e}")
            raise

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=getattr(self, "milestones", [1, 35, 40]),
            gamma=getattr(self, "lr_gamma", 0.5),
        )
        log_string(log, "======================================================")

    # -------------------- 归一化与 batch 构建 --------------------
    def _update_running_norm(self, batch_X: np.ndarray):
        """简化版在线归一化统计更新。"""
        flat = batch_X[..., 0].reshape(-1)
        batch_mean = float(flat.mean())
        batch_std = float(flat.std())
        alpha = 0.01
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.std = (1 - alpha) * self.std + alpha * max(batch_std, 1e-6)

    def _build_batch_from_records(self, records):
        """将若干 SampleRecord 构建为 batch 的 X, Y, TE（TE 仅用前 2 维）。"""
        B = len(records)
        input_len = self.input_len
        output_len = self.output_len
        N = self.node_num

        X = np.zeros((B, input_len, N, 1), dtype=np.float32)
        Y = np.zeros((B, output_len, N, 1), dtype=np.float32)
        TE = np.zeros((B, input_len, N, 2), dtype=np.float32)

        for i, rec in enumerate(records):
            x_mat = rec.input_flows.astype(np.float32)
            y_mat = rec.output_flows.astype(np.float32)

            # 时间特征：取前 input_len 步、前 2 维 (week, hour)
            te_all = rec.time_features[:input_len, :]
            te_2 = te_all[:, :2].astype(np.float32)

            X[i, :, : x_mat.shape[1], 0] = x_mat
            Y[i, :, : y_mat.shape[1], 0] = y_mat

            TE[i] = np.repeat(te_2[:, np.newaxis, :], repeats=N, axis=1)

        return X, Y, TE

    # -------------------- 训练循环（流式） --------------------
    def train(self):
        """真正 streaming 的训练：按批从 ODPS 读行，构建 batch 即训即丢。"""
        log_string(log, "======================STREAM TRAIN MODE======================")

        from odps import ODPS

        access_id = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
        secret = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        if not access_id or not secret:
            raise RuntimeError(
                "缺少 ODPS 凭证环境变量: ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET"
            )

        odps = ODPS(
            access_id,
            secret,
            project=self.odps_project,
            endpoint="http://service-corp.odps.aliyun-inc.com/api",
        )

        table = odps.get_table(self.odps_table)
        reader_kwargs = {}
        if self.ds_partition:
            reader_kwargs["partition"] = f"ds={self.ds_partition}"

        min_loss = 1e12

        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()

            with table.open_reader(**reader_kwargs) as reader:
                buffer = []

                for row in reader:
                    row_dict = {
                        "input_flows": row["input_flows"],
                        "output_flows": row["output_flows"],
                        "time_features": row["time_features"],
                        "node_pairs": row["node_pairs"],
                        "node_count": row["node_count"],
                        "flow_mean": row["flow_mean"],
                        "flow_std": row["flow_std"],
                    }
                    self.loader.append_record_from_row(row_dict)
                    buffer.append(self.loader._samples[-1])

                    if len(buffer) >= self.batch_size:
                        X_np, Y_np, TE_np = self._build_batch_from_records(buffer)
                        buffer.clear()

                        self._update_running_norm(X_np)

                        NormX = torch.from_numpy((X_np - self.mean) / self.std).float().to(self.device)
                        NormY = torch.from_numpy((Y_np - self.mean) / self.std).float().to(self.device)
                        TE = torch.from_numpy(TE_np).to(self.device)

                        self.optimizer.zero_grad()
                        y_hat = self.model(NormX, TE)
                        loss = _compute_loss(NormY, y_hat)
                        loss.backward()
                        self.optimizer.step()

                        train_l_sum += loss.detach().cpu().item()
                        batch_count += 1

                # 处理最后不足一个 batch 的残余
                if buffer:
                    X_np, Y_np, TE_np = self._build_batch_from_records(buffer)
                    buffer.clear()

                    self._update_running_norm(X_np)

                    NormX = torch.from_numpy((X_np - self.mean) / self.std).float().to(self.device)
                    NormY = torch.from_numpy((Y_np - self.mean) / self.std).float().to(self.device)
                    TE = torch.from_numpy(TE_np).to(self.device)

                    self.optimizer.zero_grad()
                    y_hat = self.model(NormX, TE)
                    loss = _compute_loss(NormY, y_hat)
                    loss.backward()
                    self.optimizer.step()

                    train_l_sum += loss.detach().cpu().item()
                    batch_count += 1

            self.lr_scheduler.step()

            avg_loss = train_l_sum / max(batch_count, 1)
            log_string(
                log,
                f"epoch {epoch}, lr {self.optimizer.param_groups[0]['lr']:.6f}, "
                f"train loss {avg_loss:.6f}, time {time.time() - start:.1f}s",
            )

            if avg_loss < min_loss:
                min_loss = avg_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_file)
                log_string(log, f"保存模型到: {self.model_file}")

        log_string(log, f"最佳 epoch: {self.best_epoch}")
        log_string(log, "======================STREAM TRAIN END======================")


# -------------------- 配置解析 --------------------

def parse_config(config_file: str) -> dict:
    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}
    # Data
    params["odps_project"] = config.get("Data", "odps_project")
    params["odps_table"] = config.get("Data", "odps_table")
    params["ds_partition"] = config.get("Data", "ds_partition")
    params["train_ratio"] = config.getfloat("Data", "train_ratio", fallback=0.7)
    params["val_ratio"] = config.getfloat("Data", "val_ratio", fallback=0.1)
    params["use_6dim_time_feat"] = config.getboolean("Data", "use_6dim_time_feat", fallback=False)

    if config.has_option("Data", "inter_location_table"):
        params["inter_location_table"] = config.get("Data", "inter_location_table")

    # Model
    params["input_len"] = config.getint("Model", "input_len", fallback=12)
    params["output_len"] = config.getint("Model", "output_len", fallback=12)
    params["tem_patchsize"] = config.getint("Model", "tem_patchsize")
    params["tem_patchnum"] = config.getint("Model", "tem_patchnum")
    params["spa_patchsize"] = config.getint("Model", "spa_patchsize")
    params["spa_patchnum"] = config.getint("Model", "spa_patchnum")
    params["layers"] = config.getint("Model", "layers")
    params["factors"] = config.getint("Model", "factors")
    params["input_dims"] = config.getint("Model", "input_dims")
    params["node_dims"] = config.getint("Model", "node_dims")
    params["recurtimes"] = config.getint("Model", "recurtimes")

    # Training
    params["learning_rate"] = config.getfloat("Training", "learning_rate")
    params["weight_decay"] = config.getfloat("Training", "weight_decay")
    params["batch_size"] = config.getint("Training", "batch_size")
    params["max_epoch"] = config.getint("Training", "max_epoch")

    # Log & Model
    params["log_file"] = config.get("Log", "log_file")
    params["model_file"] = config.get("Log", "model_file")

    # 可选：单独的 log_dir，若存在则覆盖 log_file 中的目录部分
    if config.has_option("Log", "log_dir"):
        params["log_dir"] = config.get("Log", "log_dir")

    return params


DEFAULT_CONFIG_PATH = os.path.join("config", "samples.conf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH, help="配置文件路径，默认 config/samples.conf"
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU 设备编号")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"找不到配置文件: {args.config}，如需自定义请使用 --config 指定")

    cfg = parse_config(args.config)
    cfg["cuda"] = args.cuda

    # 捕获所有异常并写入日志文件
    import traceback
    global log
    log = None
    try:
        solver = StreamSampleSolver(cfg)
        solver.train()
    except Exception as e:
        # 如果日志已经初始化成功，则把异常也写入日志
        if log is not None:
            log_string(log, "======================EXCEPTION======================")
            log_string(log, repr(e))
            log_string(log, traceback.format_exc())
            log_string(log, "====================================================")
        # 同时继续抛出，让 nohup/out 也有 traceback
        raise
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
