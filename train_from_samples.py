"""
从 MaxCompute 预处理样本表训练 PatchSTG 模型

与 main.py 的区别:
1. 数据来源: tb_patchstg_train_samples_full (预处理表) 而非 npz 文件
2. 时间特征: 6维 (week, hour, minute, day_type, day, month) 而非 2维 (tod, dow)
3. 城市范围: 混合多城市训练
4. 数据格式: 已经是滑动窗口样本,不需要 seq2instance

使用方法:
    python train_from_samples.py --config config/samples.conf --cuda 0

作者: AI Assistant
日期: 2025-11-15
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm
import os

# 引入模型与工具函数
from models.model import PatchSTG
from lib.utils import log_string, _compute_loss, metric, loadDataFromSamples
from lib.patchstg_sample_loader import PatchSTGSampleLoader, build_locations_from_pairs


class SampleSolver(object):
    """
    使用预处理样本训练 PatchSTG 的 Solver
    
    主要改动:
    - 使用 PatchSTGSampleLoader 加载数据
    - 支持 6 维时间特征
    - 模型需要适配新的时间特征维度
    """
    
    def __init__(self, config):
        """
        参数:
            config (dict): 配置参数字典
        """
        self.__dict__.update(**config)
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self._build_timestamped_log_path(self.log_file, self.run_timestamp)
        
        # 初始化日志
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        global log
        log = open(self.log_file, 'w')
        log_string(log, "======================CONFIG======================")
        for key, value in config.items():
            log_string(log, f"{key}: {value}")
        log_string(log, "==================================================")
        log_string(log, f"日志文件: {self.log_file}")
        
        # 加载数据
        self.load_sample_data()
        
        # 设备选择
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        log_string(log, f"使用设备: {self.device}")
        
        # 构建模型
        self.build_model()
        
        self.best_epoch = 0

    def _build_timestamped_log_path(self, base_path, timestamp):
        """根据启动时间生成带时间戳的日志文件路径"""
        if not base_path:
            base_path = os.path.join('log', 'patchstg.log')
        log_dir = os.path.dirname(base_path) or '.'
        filename = os.path.basename(base_path) or 'patchstg.log'
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = '.log'
        timestamped_name = f"{name}_{timestamp}{ext}"
        return os.path.join(log_dir, timestamped_name)
    
    def load_sample_data(self):
        """
        使用 PatchSTGSampleLoader 预加载一部分样本数据（小规模调试用）
        """
        log_string(log, "======================LOADING DATA======================")
        
        # 初始化数据加载器
        loader = PatchSTGSampleLoader(
            odps_project=self.odps_project,
            odps_table=self.odps_table,
            ds_partition=self.ds_partition,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            use_6dim_time_feat=getattr(self, 'use_6dim_time_feat', False),
        )
        
        # 加载样本数据
        if hasattr(self, 'debug_csv') and self.debug_csv:
            # 调试模式: 从本地CSV加载
            log_string(log, f"调试模式: 从CSV加载数据: {self.debug_csv}")
            loader.load_from_csv(self.debug_csv)
        else:
            # 生产模式: 从MaxCompute加载
            log_string(log, f"从 MaxCompute 加载数据: {self.odps_table}, ds={self.ds_partition}")
            loader.load_from_odps(limit=getattr(self, 'data_limit', None))
        
        # 加载路口位置信息
        if hasattr(self, 'inter_location_dict') and self.inter_location_dict:
            # 从字典加载 (用于调试)
            loader.load_inter_locations(self.inter_location_dict)
        elif hasattr(self, 'inter_location_table'):
            # 从 MaxCompute 表加载
            loader.load_inter_locations_from_odps(self.inter_location_table)
        else:
            log_string(log, "警告: 未提供路口位置信息,将无法进行空间划分")
        
        # 准备训练数据
        data_dict = loader.prepare_training_data(normalize=False)
        self.sample_metadata = data_dict['metadata']
        
        # 构建 locations 矩阵 (用于KDTree)
        unique_pairs = self.sample_metadata['node_pairs']
        if loader.inter_id_locations:
            locations = build_locations_from_pairs(unique_pairs, loader.inter_id_locations)
            log_string(log, f"构建 locations 矩阵: shape={locations.shape}")
        else:
            # 如果没有位置信息,使用随机位置 (仅用于调试)
            log_string(log, "警告: 使用随机位置进行空间划分 (仅调试)")
            num_pairs = len(unique_pairs)
            locations = np.random.rand(2, num_pairs).astype(np.float32)
        
        # 使用 loadDataFromSamples 进行空间划分
        adjpath = os.path.join(self.model_file.replace('.pth', '_adj.npy'))
        result = loadDataFromSamples(
            data_dict,
            locations,
            adjpath,
            self.recurtimes,
            self.spa_patchsize,
            log
        )
        
        # 解包结果
        (self.trainX, self.trainY, self.trainXTE, self.trainYTE,
         self.valX, self.valY, self.valXTE, self.valYTE,
         self.testX, self.testY, self.testXTE, self.testYTE,
         self.mean, self.std,
         self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx) = result
        
        # 更新节点数量 (实际的最大节点数)
        self.node_num = self.trainX.shape[2]
        log_string(log, f"节点数量 (max_nodes): {self.node_num}")
        
        # 时间特征维度
        self.time_feat_dim = self.sample_metadata.get('time_feat_dim', 2)
        log_string(log, f"时间特征维度: {self.time_feat_dim}")
        log_string(log, "=======================================================")
    
    def build_model(self):
        """
        构建模型、优化器与学习率调度器
        
        注意: 需要修改模型以支持6维时间特征
        """
        log_string(log, "======================BUILD MODEL======================")
        
        # 实例化模型
        # 注意: 原始 PatchSTG 假设时间特征是 2 维 (tod, dow)
        # 现在需要传入 6 维特征,可能需要修改模型代码
        # 这里先使用原始模型,后续可能需要调整
        
        try:
            self.model = PatchSTG(
                self.output_len,
                self.tem_patchsize,
                self.tem_patchnum,
                self.node_num,
                self.spa_patchsize,
                self.spa_patchnum,
                # 时间特征相关 - 保持兼容性
                self.tod if hasattr(self, 'tod') else 24,  # tod 默认 24
                self.dow if hasattr(self, 'dow') else 7,   # dow 默认 7
                self.layers,
                self.factors,
                self.input_dims,
                self.node_dims,
                # 时间嵌入维度 - 需要根据6维调整
                getattr(self, 'tod_dims', 8),  # 可以复用或调整
                getattr(self, 'dow_dims', 8),
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx
            ).to(self.device)
            
            log_string(log, f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        except Exception as e:
            log_string(log, f"模型构建失败: {e}")
            log_string(log, "可能需要修改 PatchSTG 模型以支持 6 维时间特征")
            raise
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=getattr(self, 'milestones', [1, 35, 40]),
            gamma=getattr(self, 'lr_gamma', 0.5),
        )
        
        log_string(log, "=======================================================")
    
    def vali(self):
        """验证模式"""
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []
        
        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
                
                X = self.valX[start_idx : end_idx]
                Y = self.valY[start_idx : end_idx]
                TE = torch.from_numpy(self.valXTE[start_idx : end_idx]).to(self.device)
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                
                y_hat = self.model(NormX, TE)
                
                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(Y)
        
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        
        # 计算指标
        maes, rmses, mapes = [], [], []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, f'step {i+1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, f'average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
    
    def train(self):
        """训练模式"""
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]
        
        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()
            
            # 随机打乱训练数据
            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainXTE = self.trainXTE[permutation]
            self.trainYTE = self.trainYTE[permutation]
            
            num_batch = math.ceil(num_train / self.batch_size)
            
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_train, (batch_idx + 1) * self.batch_size)
                
                X = self.trainX[start_idx : end_idx]
                Y = self.trainY[start_idx : end_idx]
                TE = torch.from_numpy(self.trainXTE[start_idx : end_idx]).to(self.device)
                
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                NormY = torch.from_numpy((Y - self.mean) / self.std).float().to(self.device)
                
                self.optimizer.zero_grad()
                y_hat = self.model(NormX, TE)
                loss = _compute_loss(NormY, y_hat)
                
                loss.backward()
                self.optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
            
            self.lr_scheduler.step()
            
            # 验证
            log_string(log, f'epoch {epoch}, lr {self.optimizer.param_groups[0]["lr"]:.6f}, '
                           f'train loss {train_l_sum / batch_count:.6f}, time {time.time() - start:.1f}s')
            
            maes, rmses, mapes = self.vali()
            
            # 保存最优模型
            if maes[-1] < min_loss:
                min_loss = maes[-1]
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_file)
                log_string(log, f'保存模型到: {self.model_file}')
        
        log_string(log, f'最佳 epoch: {self.best_epoch}')
        log_string(log, "======================TRAIN END======================")
    
    def test(self):
        """测试模式"""
        log_string(log, "======================TEST MODE======================")
        
        # 加载最优模型
        self.model.load_state_dict(torch.load(self.model_file))
        log_string(log, f'加载模型: {self.model_file}')
        
        self.model.eval()
        num_test = self.testX.shape[0]
        pred = []
        label = []
        
        num_batch = math.ceil(num_test / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_test, (batch_idx + 1) * self.batch_size)
                
                X = self.testX[start_idx : end_idx]
                Y = self.testY[start_idx : end_idx]
                TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                
                y_hat = self.model(NormX, TE)
                
                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(Y)
        
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        
        # 计算指标
        maes, rmses, mapes = [], [], []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, f'step {i+1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        mae, rmse, mape = metric(pred, label)
        log_string(log, f'average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        log_string(log, "=======================================================")


class StreamSampleSolver(SampleSolver):
    """真正流式的训练 Solver：按批从 ODPS 读样本，即训即丢，不在内存中保留全量 X/Y。

    设计要点：
    - 仍然在构造函数里完成日志初始化、设备选择、位置加载和 KDTree 构建；
    - 但不再通过 loadDataFromSamples 生成全量 trainX/valX/testX；
    - 训练阶段：直接使用 PatchSTGSampleLoader.append_record_from_row 按行解析，
      累积到一个 batch，再做一次前向+反向，然后清空 batch。
    - 时间特征：从 6 维中取前 2 维，对应原模型 2 维时间嵌入。
    """

    def __init__(self, config):
        # 先完成基础初始化（日志、配置、device 等）
        self.__dict__.update(**config)
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self._build_timestamped_log_path(self.log_file, self.run_timestamp)

        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        global log
        log = open(self.log_file, 'w')
        log_string(log, "======================CONFIG======================")
        for key, value in config.items():
            log_string(log, f"{key}: {value}")
        log_string(log, "==================================================")
        log_string(log, f"日志文件: {self.log_file}")

        # 初始化 loader，但不一次性加载所有样本
        self.loader = PatchSTGSampleLoader(
            odps_project=self.odps_project,
            odps_table=self.odps_table,
            ds_partition=self.ds_partition,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            use_6dim_time_feat=getattr(self, 'use_6dim_time_feat', False),
        )

        # 加载路口位置信息，构建 locations + KDTree（只需跑一次）
        self._init_spatial_structure()

        # 设备和模型
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        log_string(log, f"使用设备: {self.device}")

        self.build_model()
        self.best_epoch = 0

        # 初始化归一化统计（运行时更新）
        self.mean = 0.0
        self.std = 1.0
        self._norm_count = 0

    def _init_spatial_structure(self):
        """读取一小部分样本，推断 node_pairs，构建 locations 和 KDTree 索引。"""
        log_string(log, "==================INIT SPATIAL STRUCTURE==================")

        # 先用 limit 读取少量样本到 _samples
        self.loader.load_from_odps(limit=getattr(self, 'init_sample_limit', 1024))
        if not self.loader._samples:
            raise RuntimeError("初始化失败：未从样本表中读取到任何样本")

        # 使用 prepare_training_data 构建一次 data_dict 和 KDTree，拿到索引结构
        data_dict = self.loader.prepare_training_data(normalize=False)
        metadata = data_dict['metadata']
        self.sample_metadata = metadata

        unique_pairs = metadata['node_pairs']

        # 加载路口位置信息
        if hasattr(self, 'inter_location_dict') and self.inter_location_dict:
            self.loader.load_inter_locations(self.inter_location_dict)
        elif hasattr(self, 'inter_location_table'):
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

        # 跑一遍 loadDataFromSamples 只为拿到 KDTree 索引
        adjpath = os.path.join(self.model_file.replace('.pth', '_adj.npy'))
        (trainX, trainY, trainXTE, trainYTE,
         valX, valY, valXTE, valYTE,
         testX, testY, testXTE, testYTE,
         mean, std,
         self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx) = loadDataFromSamples(
            data_dict,
            locations,
            adjpath,
            self.recurtimes,
            self.spa_patchsize,
            log
        )

        # 基于 init 样本的节点数量
        self.node_num = trainX.shape[2]
        log_string(log, f"节点数量 (max_nodes): {self.node_num}")

        # 初始化 mean/std 为 init 样本的统计量，后续训练过程中可以继续更新
        self.mean = float(mean)
        self.std = float(std if std != 0 else 1.0)
        self._norm_count = trainX.shape[0]

        log_string(log, f"初始 Mean: {self.mean}, Std: {self.std}")
        log_string(log, "=======================================================")

    def _update_running_norm(self, batch_X: np.ndarray):
        """在线更新 mean/std（简化版），避免一次性加载全量数据。"""
        # batch_X: (B, T, N, 1)
        batch = batch_X[..., 0].reshape(-1)
        batch_mean = float(batch.mean())
        batch_std = float(batch.std())

        # 简单的加权更新（可以换成更严谨的 Welford 算法）
        alpha = 0.01
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.std = (1 - alpha) * self.std + alpha * max(batch_std, 1e-6)

    def _build_batch_from_records(self, records):
        """将若干 SampleRecord 构建为一个 batch 的 X, Y, TE。

        - X: (B, input_len, node_num, 1)
        - Y: (B, output_len, node_num, 1)
        - TE: (B, input_len, node_num, 2)  只取 6 维中的前 2 维
        """
        B = len(records)
        input_len = self.input_len
        output_len = self.output_len
        N = self.node_num

        X = np.zeros((B, input_len, N, 1), dtype=np.float32)
        Y = np.zeros((B, output_len, N, 1), dtype=np.float32)
        TE = np.zeros((B, input_len, N, 2), dtype=np.float32)

        for i, rec in enumerate(records):
            # rec.input_flows: (input_len, node_count)
            # rec.output_flows: (output_len, node_count)
            # 这里假设 node_count == self.node_num
            x_mat = rec.input_flows.astype(np.float32)
            y_mat = rec.output_flows.astype(np.float32)

            # 时间特征：取前 input_len 步、前 2 维 [week, hour]
            te_all = rec.time_features[:input_len, :]
            te_2 = te_all[:, :2].astype(np.float32)

            # 填入 batch，假设节点顺序与 KDTree 中一致
            X[i, :, :x_mat.shape[1], 0] = x_mat
            Y[i, :, :y_mat.shape[1], 0] = y_mat

            # TE broadcast 到节点维度
            TE[i] = np.repeat(te_2[:, np.newaxis, :], repeats=N, axis=1)

        return X, Y, TE

    def train(self):
        """真正 streaming 的训练：按批从 ODPS 读行，构建 batch 即训即丢。"""
        log_string(log, "======================STREAM TRAIN MODE======================")

        from odps import ODPS

        access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
        if not access_id or not secret:
            raise RuntimeError("缺少 ODPS 凭证环境变量: ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET")

        project = self.odps_project
        table_name = self.odps_table

        odps = ODPS(
            access_id,
            secret,
            project=project,
            endpoint='http://service-corp.odps.aliyun-inc.com/api'
        )

        table = odps.get_table(table_name)
        reader_kwargs = {}
        if self.ds_partition:
            reader_kwargs["partition"] = f"ds={self.ds_partition}"

        min_loss = 1e12

        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()

            # 每个 epoch 重新遍历一次表（对大表可能需要按 ds/时间再细分）
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

                        # 在线更新 mean/std
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
            log_string(log, f'epoch {epoch}, lr {self.optimizer.param_groups[0]["lr"]:.6f}, '
                               f'train loss {avg_loss:.6f}, time {time.time() - start:.1f}s')

            # 简化：streaming 版本先不做 val/test，后续可以按天抽样一部分样本做验证
            if avg_loss < min_loss:
                min_loss = avg_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_file)
                log_string(log, f'保存模型到: {self.model_file}')

        log_string(log, f'最佳 epoch: {self.best_epoch}')
        log_string(log, "======================STREAM TRAIN END======================")


def parse_config(config_file):
    """解析配置文件"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 提取配置参数
    params = {}
    
    # Data
    params['odps_project'] = config.get('Data', 'odps_project')
    params['odps_table'] = config.get('Data', 'odps_table')
    params['ds_partition'] = config.get('Data', 'ds_partition')
    params['train_ratio'] = config.getfloat('Data', 'train_ratio', fallback=0.7)
    params['val_ratio'] = config.getfloat('Data', 'val_ratio', fallback=0.1)
    params['use_6dim_time_feat'] = config.getboolean('Data', 'use_6dim_time_feat', fallback=False)
    
    if config.has_option('Data', 'inter_location_table'):
        params['inter_location_table'] = config.get('Data', 'inter_location_table')
    
    if config.has_option('Data', 'debug_csv'):
        params['debug_csv'] = config.get('Data', 'debug_csv')
    
    # Model
    params['input_len'] = config.getint('Model', 'input_len', fallback=12)
    params['output_len'] = config.getint('Model', 'output_len', fallback=12)
    params['tem_patchsize'] = config.getint('Model', 'tem_patchsize')
    params['tem_patchnum'] = config.getint('Model', 'tem_patchnum')
    params['spa_patchsize'] = config.getint('Model', 'spa_patchsize')
    params['spa_patchnum'] = config.getint('Model', 'spa_patchnum')
    params['layers'] = config.getint('Model', 'layers')
    params['factors'] = config.getint('Model', 'factors')
    params['input_dims'] = config.getint('Model', 'input_dims')
    params['node_dims'] = config.getint('Model', 'node_dims')
    params['recurtimes'] = config.getint('Model', 'recurtimes')
    
    # Training
    params['learning_rate'] = config.getfloat('Training', 'learning_rate')
    params['weight_decay'] = config.getfloat('Training', 'weight_decay')
    params['batch_size'] = config.getint('Training', 'batch_size')
    params['max_epoch'] = config.getint('Training', 'max_epoch')
    
    # Log
    params['log_file'] = config.get('Log', 'log_file')
    params['model_file'] = config.get('Log', 'model_file')
    
    return params


DEFAULT_CONFIG_PATH = os.path.join('config', 'samples.conf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='配置文件路径，默认 config/samples.conf')
    parser.add_argument('--cuda', type=int, default=0, help='GPU 设备编号')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--stream', action='store_true', help='是否使用真正 streaming 训练循环')
    
    args = parser.parse_args()
    
    # 解析配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"找不到配置文件: {args.config}，如需自定义请使用 --config 指定")

    config = parse_config(args.config)
    config['cuda'] = args.cuda
    
    # 创建 Solver：根据 --stream 切换实现
    if args.stream:
        solver = StreamSampleSolver(config)
    else:
        solver = SampleSolver(config)
    
    # 运行
    if args.mode == 'train':
        solver.train()
        # streaming 版本暂不做 test，后续可扩展
        if not args.stream:
            solver.test()
    else:
        solver.test()
    
    log.close()
