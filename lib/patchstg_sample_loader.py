import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Iterable

import numpy as np

try:
    # 与 odps_table_data_loader 保持一致，优先使用 pyodps
    from odps import ODPS
except ImportError:  # 本地开发环境可能没有安装
    ODPS = None


"""PatchSTG 样本表加载器

当前版本做一件事：
- 不直接连 MaxCompute，而是假设你已经从
  `autonavi_traffic_report.tb_patchstg_train_samples_full`
  导出了一个本地 JSON/NDJSON/CSV 文件，或者未来再补 ODPS 读表逻辑。

先保证 `train_from_samples.py` 能在本地跑起来，
后续如果你要我接 ODPS SDK，我们再扩展这里。

核心输出：
  data_dict = {
      'train': {'X': ..., 'Y': ..., 'TE': ...},
      'val':   {'X': ..., 'Y': ..., 'TE': ...},
      'test':  {'X': ..., 'Y': ..., 'TE': ...},
      'metadata': {
          'mean': float,
          'std': float,
          'node_pairs': List[Tuple[int, int]],
          'node_num': int,
          'time_feat_dim': int,
      },
  }

注意：
- 这里把样本表的一条记录视为一个滑窗样本，
  `input_flows` → X 的 12 步流量，`output_flows` → Y 的 12 步流量。
- `time_features` 拆成 (input_len + output_len, 6) 然后切成
  XTE: 前 input_len 步，YTE: 后 output_len 步。
- 为了和旧代码兼容，TE 只返回 X 对应的时间特征，YTE 在
  utils.loadDataFromSamples 里通过 copy 得到。
"""


@dataclass
class SampleRecord:
    input_flows: np.ndarray   # (input_len, node_num)
    output_flows: np.ndarray  # (output_len, node_num)
    time_features: np.ndarray # (input_len + output_len, time_feat_dim)
    node_pairs: np.ndarray    # (node_num, 2)  每个节点的 (up_inter_id, down_inter_id)
    flow_mean: float
    flow_std: float


class PatchSTGSampleLoader:
    def __init__(
        self,
        odps_project: str = "",
        odps_table: str = "",
        ds_partition: str = "",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        use_6dim_time_feat: bool = False,
    ) -> None:
        self.odps_project = odps_project
        self.odps_table = odps_table
        self.ds_partition = ds_partition
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_6dim_time_feat = use_6dim_time_feat

        # inter_id -> (lat, lng)
        self.inter_id_locations: Dict[str, Tuple[float, float]] = {}

        # 内部缓存的样本列表（仅用于非流式或小规模调试）
        self._samples: List[SampleRecord] = []

    # ====== 数据加载入口：从 ODPS 表流式读取样本 ======
    def load_from_odps(self, limit: Optional[int] = None) -> None:
        """直接连接 ODPS，读取 autonavi_traffic_report.tb_patchstg_train_samples_full。

        这里采用和 `odps_table_data_loader` 同样的方式，从环境变量
        `ALIBABA_CLOUD_ACCESS_KEY_ID` 和 `ALIBABA_CLOUD_ACCESS_KEY_SECRET` 中读取凭证，
        endpoint 使用企业内网地址 `http://service-corp.odps.aliyun-inc.com/api`。

        注意：
        - 本函数会把读取到的样本全部放到 self._samples 中，适合小规模调试；
        - 真正大规模流式训练时，建议你在训练循环里直接使用 append_record_from_row
          对 row 做解析，然后即时训练并丢弃，不一定要保存在 _samples 里。
        """
        if ODPS is None:
            raise RuntimeError("未安装 pyodps，请在远程环境中安装 odps 包后再运行。")

        access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

        if not access_id or not secret:
            raise RuntimeError("缺少 ODPS 凭证环境变量: ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET")

        project = self.odps_project or 'autonavi_traffic_report'
        table_name = self.odps_table or 'tb_patchstg_train_samples_full'

        odps = ODPS(
            access_id,
            secret,
            project=project,
            endpoint='http://service-corp.odps.aliyun-inc.com/api'
        )

        table = odps.get_table(table_name)

        # 如果有 ds 分区，则加上
        reader_kwargs = {}
        if self.ds_partition:
            reader_kwargs["partition"] = f"ds={self.ds_partition}"

        cnt = 0
        self._samples.clear()

        with table.open_reader(**reader_kwargs) as reader:
            for row in reader:  # type: ignore
                row_dict = {
                    "input_flows": row["input_flows"],
                    "output_flows": row["output_flows"],
                    "time_features": row["time_features"],
                    "node_pairs": row["node_pairs"],
                    "node_count": row["node_count"],
                    "flow_mean": row["flow_mean"],
                    "flow_std": row["flow_std"],
                }
                self.append_record_from_row(row_dict)
                cnt += 1
                if limit is not None and cnt >= limit:
                    break

        # 此时 self._samples 中已经有若干 SampleRecord，可用于非流式训练或调试

    def load_from_csv(self, csv_path: str) -> None:
        """占位：如果你后续把样本导成 CSV，这里再实现。

        先抛错，避免你误以为已经实现了。
        """
        raise NotImplementedError(
            "load_from_csv 尚未实现，请先使用 JSON/NDJSON 或扩展此函数。"
        )

    # ====== 位置加载（交给 train_from_samples.py 调用） ======
    def load_inter_locations(self, inter_location_dict: Dict[str, Tuple[float, float]]) -> None:
        """直接使用传入的路口经纬度字典。"""
        self.inter_id_locations = dict(inter_location_dict)

    def load_inter_locations_from_odps(self, inter_location_table: str) -> None:
        """从 ODPS 路口表 autonavi_traffic_report.intersection_meta_1 加载经纬度。

        复用与 `odps_table_data_loader._load_location_dict` 相同的 ODPS 连接方式，
        使用环境变量中的 AK/SK 和企业内网 endpoint。
        """
        if ODPS is None:
            raise RuntimeError("未安装 pyodps，请在远程环境中安装 odps 包后再运行。")

        access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

        if not access_id or not secret:
            raise RuntimeError("缺少 ODPS 凭证环境变量: ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET")

        project = self.odps_project or 'autonavi_traffic_report'
        meta_table = inter_location_table or 'intersection_meta_1'

        odps = ODPS(
            access_id,
            secret,
            project=project,
            endpoint='http://service-corp.odps.aliyun-inc.com/api'
        )

        # 这里只需要 inter_id / lat / lng 即可
        query = f"""
        SELECT inter_id, lat, lng
        FROM {project}.{meta_table}
        WHERE 1 = 1
        """

        self.inter_id_locations = {}

        with odps.execute_sql(query).open_reader() as reader:  # type: ignore
            for record in reader:
                inter_id = record[0]
                lat = record[1]
                lng = record[2]
                if inter_id is None or lat is None or lng is None:
                    continue
                self.inter_id_locations[str(inter_id)] = (float(lat), float(lng))

    # ====== 主功能：把内部样本列表组装成 PatchSTG 需要的数据结构 ======
    def prepare_training_data(self, normalize: bool = False) -> Dict[str, Any]:
        if not self._samples:
            raise RuntimeError("尚未加载任何样本，请先调用 load_from_odps/load_from_csv 等方法。")

        # 以第一条样本为“模板”，其他样本如果长度或节点数不一致，就跳过
        ref = self._samples[0]
        input_len = ref.input_flows.shape[0]
        output_len = ref.output_flows.shape[0]
        node_num = ref.input_flows.shape[1]
        time_feat_dim = ref.time_features.shape[1]

        valid_samples: List[SampleRecord] = []
        for s in self._samples:
            if (
                s.input_flows.shape[0] == input_len
                and s.output_flows.shape[0] == output_len
                and s.input_flows.shape[1] == node_num
                and s.output_flows.shape[1] == node_num
                and s.time_features.shape[0] >= input_len
            ):
                valid_samples.append(s)

        if not valid_samples:
            raise RuntimeError("prepare_training_data: 所有样本的形状都与第一条不一致，无法构建训练数据")

        num_samples = len(valid_samples)

        X = np.stack([s.input_flows for s in valid_samples], axis=0)  # (N, input_len, node_num)
        Y = np.stack([s.output_flows for s in valid_samples], axis=0) # (N, output_len, node_num)

        # 只取前 input_len 步作为 X 的时间特征
        TE = np.stack(
            [s.time_features[:input_len] for s in valid_samples],
            axis=0,
        )  # (N, input_len, time_feat_dim)

        # 扩展流量到 (N, T, N_nodes, 1)
        X = X[..., np.newaxis].astype(np.float32)
        Y = Y[..., np.newaxis].astype(np.float32)

        # 时间特征扩展到 (N, T, N_nodes, time_feat_dim)
        TE = np.repeat(TE[:, :, np.newaxis, :], repeats=node_num, axis=2).astype(np.float32)

    # 统一 node_pairs（按模板样本）
    node_pairs = ref.node_pairs  # (node_num, 2)

        # 计算 mean/std（或者使用样本里的 flow_mean / flow_std）
        if normalize:
            mean = float(np.mean(X))
            std = float(np.std(X) + 1e-6)
        else:
            # 如果每条样本的 mean/std 一致，可以直接取第一条
            mean = float(ref.flow_mean)
            std = float(ref.flow_std)

        # 划分 train/val/test
        n_train = int(num_samples * self.train_ratio)
        n_val = int(num_samples * self.val_ratio)
        n_test = num_samples - n_train - n_val
        if n_test <= 0:
            n_test = max(1, num_samples - n_train - n_val)

        train_slice = slice(0, n_train)
        val_slice = slice(n_train, n_train + n_val)
        test_slice = slice(n_train + n_val, num_samples)

        data_dict = {
            'train': {
                'X': X[train_slice],
                'Y': Y[train_slice],
                'TE': TE[train_slice],
            },
            'val': {
                'X': X[val_slice],
                'Y': Y[val_slice],
                'TE': TE[val_slice],
            },
            'test': {
                'X': X[test_slice],
                'Y': Y[test_slice],
                'TE': TE[test_slice],
            },
            'metadata': {
                'mean': mean,
                'std': std,
                'node_pairs': node_pairs,
                'node_num': int(node_num),
                'time_feat_dim': int(time_feat_dim),
            },
        }

        return data_dict

    # ====== 辅助：从一行 ODPS 记录解析出 SampleRecord ======
    def append_record_from_row(self, row: Dict[str, Any]) -> None:
        """给定一行 ODPS 记录（dict 风格），解析并追加到 self._samples。

        期望字段：
          - input_flows:  "row;row;..."，每个 row 是用逗号分隔的 node 流量
          - output_flows: 同上
          - time_features: "w h m day_type day month;..." 共 input_len+output_len 行
          - node_pairs:    "up,down;up,down;..."，长度 = node_count
          - node_count:    bigint
          - flow_mean, flow_std: double
        """
        # 解析 node_count
        node_count = int(row["node_count"])

        # 解析 input_flows / output_flows
        def parse_flows(s: str) -> np.ndarray:
            # 每一行: "v1,v2,...,vN"，行与行之间用 ';' 分割
            rows = [r for r in s.split(";") if r]
            mat = []
            for r in rows:
                vals = [float(x) for x in r.split(",") if x]
                # 对齐 node_count，避免偶发缺失
                if len(vals) < node_count:
                    vals += [0.0] * (node_count - len(vals))
                elif len(vals) > node_count:
                    vals = vals[:node_count]
                mat.append(vals)
            return np.array(mat, dtype=np.float32)

        input_flows = parse_flows(row["input_flows"])
        output_flows = parse_flows(row["output_flows"])

        input_len = input_flows.shape[0]
        output_len = output_flows.shape[0]

        # 解析 time_features: "w h m day_type day month;..."
        def parse_time_features(s: str) -> np.ndarray:
            rows = [r for r in s.split(";") if r]
            feat_mat = []
            for r in rows:
                parts = [p for p in r.split(" ") if p]
                # 期望 6 维: week, hour, minute, day_type, day, month
                vals = [float(x) for x in parts]
                feat_mat.append(vals)
            return np.array(feat_mat, dtype=np.float32)

        time_features = parse_time_features(row["time_features"])  # (input_len + output_len, 6)

        # 解析 node_pairs: "u,v;u,v;..."
        def parse_node_pairs(s: str) -> np.ndarray:
            pairs = []
            for seg in s.split(";"):
                seg = seg.strip()
                if not seg:
                    continue
                up_down = [p for p in seg.split(",") if p]
                if len(up_down) != 2:
                    continue
                up, down = up_down
                pairs.append((int(up), int(down)))
            return np.array(pairs, dtype=np.int64)

        node_pairs = parse_node_pairs(row["node_pairs"])

        flow_mean = float(row.get("flow_mean", 0.0))
        flow_std = float(row.get("flow_std", 1.0) or 1.0)

        record = SampleRecord(
            input_flows=input_flows,
            output_flows=output_flows,
            time_features=time_features,
            node_pairs=node_pairs,
            flow_mean=flow_mean,
            flow_std=flow_std,
        )

        self._samples.append(record)


def build_locations_from_pairs(node_pairs: np.ndarray, inter_id_locations: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """根据 (up_inter_id, down_inter_id) 对，构建每个转向的空间坐标。

    简化策略：
    - 对于每一对 (u, v)，取 u/v 的经纬度做平均，得到该转向的代表位置。
    - inter_id_locations 形如 {"up_inter_id": (lat, lng), ...}

    返回值:
        locations: np.ndarray, shape = (2, num_pairs)
    """
    num_pairs = node_pairs.shape[0]
    locs = np.zeros((2, num_pairs), dtype=np.float32)

    for i, (u, v) in enumerate(node_pairs):
        u_str = str(u)
        v_str = str(v)
        u_loc = inter_id_locations.get(u_str, (0.0, 0.0))
        v_loc = inter_id_locations.get(v_str, (0.0, 0.0))
        lat = (float(u_loc[0]) + float(v_loc[0])) / 2.0
        lng = (float(u_loc[1]) + float(v_loc[1])) / 2.0
        locs[0, i] = lat
        locs[1, i] = lng

    return locs
