"""
ODPS 预处理数据加载器
使用预处理后的时间序列表：tb_inter_traffic_timeseries

数据格式：
- 每条记录 = 一个时间窗口的所有节点数据
- flow_matrix_sparse: "0:15;1:8;2:0;..." (节点索引:流量值)
- history_matrix_sparse: "0:5;3;2;...;1:7;8;..." (节点索引:历史序列)
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from lib.utils import log_string


class PreprocessedODPSDataset(Dataset):
    """预处理后的 ODPS 数据集"""
    
    def __init__(self, 
                 odps_instance,
                 table_name: str,
                 meta_table_name: str,
                 adcode: str,
                 start_date: str,
                 end_date: str,
                 input_len: int = 12,
                 output_len: int = 12,
                 log=None):
        """
        Args:
            odps_instance: ODPS 连接实例
            table_name: 预处理后的时间序列表名（tb_inter_traffic_timeseries）
            meta_table_name: 节点元数据表名（tb_inter_node_metadata）
            adcode: 城市代码
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            input_len: 输入序列长度
            output_len: 输出序列长度
            log: 日志函数
        """
        self.odps = odps_instance
        self.table_name = table_name
        self.meta_table_name = meta_table_name
        self.adcode = adcode
        self.start_date = start_date
        self.end_date = end_date
        self.input_len = input_len
        self.output_len = output_len
        self.log = log or print
        
        # 加载节点元数据
        self._load_node_metadata()
        
        # 加载时间序列数据
        self._load_timeseries_data()
        
        log_string(self.log, f"✅ 数据集初始化完成:")
        log_string(self.log, f"   - 节点数: {self.num_nodes}")
        log_string(self.log, f"   - 时间窗口数: {len(self.time_windows)}")
        log_string(self.log, f"   - 样本数: {len(self)}")
        log_string(self.log, f"   - 输入长度: {self.input_len}, 输出长度: {self.output_len}")
        
    def _load_node_metadata(self):
        """加载节点元数据"""
        log_string(self.log, f"📖 加载节点元数据: {self.meta_table_name}")
        
        query = f"""
        SELECT 
            node_idx,
            nds_id,
            next_nds_id,
            inter_id,
            lat,
            lng
        FROM {self.meta_table_name}
        ORDER BY node_idx
        """
        
        self.node_metadata = []
        self.node_locations = []
        
        with self.odps.execute_sql(query).open_reader() as reader:
            for record in reader:
                node_idx = record['node_idx']
                lat = record['lat']
                lng = record['lng']
                
                self.node_metadata.append({
                    'node_idx': node_idx,
                    'nds_id': record['nds_id'],
                    'next_nds_id': record['next_nds_id'],
                    'inter_id': record['inter_id']
                })
                
                # 位置信息（用于计算距离矩阵）
                if lat is not None and lng is not None:
                    self.node_locations.append([lat, lng])
                else:
                    self.node_locations.append([0.0, 0.0])  # 默认值
        
        self.num_nodes = len(self.node_metadata)
        self.node_locations = np.array(self.node_locations)
        
        # 统计位置覆盖率
        valid_locations = np.sum((self.node_locations != 0).any(axis=1))
        coverage = valid_locations / self.num_nodes * 100
        
        log_string(self.log, f"   ✅ 节点元数据加载完成: {self.num_nodes} 个节点")
        log_string(self.log, f"   📍 位置覆盖率: {coverage:.1f}%")
        
        assert self.num_nodes > 0, "节点数量为0"
        
    def _load_timeseries_data(self):
        """加载时间序列数据"""
        log_string(self.log, f"📖 加载时间序列数据: {self.table_name}")
        log_string(self.log, f"   - 城市: {self.adcode}")
        log_string(self.log, f"   - 日期范围: {self.start_date} ~ {self.end_date}")
        
        query = f"""
        SELECT 
            time_window,
            time_features,
            node_count,
            flow_matrix_sparse,
            history_matrix_sparse
        FROM {self.table_name}
        WHERE adcode = '{self.adcode}'
          AND ds >= '{self.start_date}'
          AND ds <= '{self.end_date}'
        ORDER BY time_window
        """
        
        self.time_windows = []
        self.flow_matrices = []
        self.history_matrices = []
        self.time_features = []
        
        with self.odps.execute_sql(query).open_reader() as reader:
            for record in reader:
                time_window = record['time_window']
                node_count = record['node_count']
                
                # 解析稀疏流量矩阵
                flow_dict = {}
                if record['flow_matrix_sparse']:
                    for item in record['flow_matrix_sparse'].split(';'):
                        if ':' in item:
                            idx, flow = item.split(':')
                            flow_dict[int(idx)] = float(flow)
                
                # 构建密集流量矩阵 (N, 1)
                flow_matrix = np.zeros((self.num_nodes, 1), dtype=np.float32)
                for idx, flow in flow_dict.items():
                    if idx < self.num_nodes:
                        flow_matrix[idx, 0] = flow
                
                # 解析稀疏历史矩阵
                history_dict = {}
                if record['history_matrix_sparse']:
                    for item in record['history_matrix_sparse'].split(';'):
                        if ':' in item:
                            parts = item.split(':')
                            if len(parts) == 2:
                                idx_str, history_str = parts
                                idx = int(idx_str)
                                # history_str 格式: "5;3;2;1;0;0;8;15;12;10;8;6"
                                history_values = [float(v) for v in history_str.split(';')]
                                history_dict[idx] = history_values
                
                # 构建密集历史矩阵 (T, N, 1)
                history_matrix = np.zeros((self.input_len, self.num_nodes, 1), dtype=np.float32)
                for idx, history in history_dict.items():
                    if idx < self.num_nodes:
                        # 取最后 input_len 个值
                        history = history[-self.input_len:]
                        for t in range(len(history)):
                            history_matrix[t, idx, 0] = history[t]
                
                # 解析时间特征
                if record['time_features']:
                    time_feat = [float(x) for x in record['time_features'].split()]
                    # 取 [day_of_week, hour]（原始代码格式）
                    if len(time_feat) >= 5:
                        day_of_week = time_feat[4]  # day_of_week
                        hour = time_feat[1]  # hour
                        time_feat_vec = [day_of_week, hour]
                    else:
                        time_feat_vec = [0, 0]
                else:
                    time_feat_vec = [0, 0]
                
                self.time_windows.append(time_window)
                self.flow_matrices.append(flow_matrix)
                self.history_matrices.append(history_matrix)
                self.time_features.append(time_feat_vec)
        
        log_string(self.log, f"   ✅ 时间序列数据加载完成: {len(self.time_windows)} 个时间窗口")
        
        # 数据验证
        assert len(self.time_windows) > 0, "时间窗口数量为0"
        assert len(self.time_windows) == len(self.flow_matrices), "数据长度不匹配"
        
        # 统计流量信息
        all_flows = np.concatenate([fm.flatten() for fm in self.flow_matrices])
        non_zero = all_flows[all_flows > 0]
        if len(non_zero) > 0:
            log_string(self.log, f"   📊 流量统计:")
            log_string(self.log, f"      - 非零值比例: {len(non_zero)/len(all_flows)*100:.2f}%")
            log_string(self.log, f"      - 流量范围: [{non_zero.min():.2f}, {non_zero.max():.2f}]")
            log_string(self.log, f"      - 平均流量: {non_zero.mean():.2f}")
    
    def __len__(self):
        """
        样本数 = 时间窗口数 - (input_len + output_len - 1)
        需要有足够的连续时间窗口来构建输入和输出序列
        """
        return max(0, len(self.time_windows) - self.input_len - self.output_len + 1)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            X: (input_len, N, 1) - 输入序列
            Y: (output_len, N, 1) - 输出序列（标签）
            TE: (input_len, 2) - 时间特征
        """
        # 输入序列：从 idx 开始的 input_len 个时间窗口
        X = np.stack([self.flow_matrices[i] for i in range(idx, idx + self.input_len)], axis=0)
        
        # 输出序列：从 idx + input_len 开始的 output_len 个时间窗口
        Y = np.stack([self.flow_matrices[i] for i in range(idx + self.input_len, idx + self.input_len + self.output_len)], axis=0)
        
        # 时间特征：输入序列的时间特征
        TE = np.array([self.time_features[i] for i in range(idx, idx + self.input_len)], dtype=np.float32)
        
        # 验证形状
        assert X.shape == (self.input_len, self.num_nodes, 1), f"X shape: {X.shape}"
        assert Y.shape == (self.output_len, self.num_nodes, 1), f"Y shape: {Y.shape}"
        assert TE.shape == (self.input_len, 2), f"TE shape: {TE.shape}"
        
        return {
            'X': X,
            'Y': Y,
            'TE': TE,
            'time_window': self.time_windows[idx]
        }


def create_preprocessed_dataloader(config, log=None):
    """
    创建预处理数据的 DataLoader
    
    Args:
        config: 配置字典，包含:
            - odps_project: ODPS 项目名
            - odps_endpoint: ODPS endpoint
            - odps_table: 时间序列表名
            - odps_meta_table: 元数据表名
            - adcode: 城市代码
            - start_date: 开始日期
            - end_date: 结束日期
            - batch_size: batch 大小
            - num_workers: 数据加载线程数
            - input_len: 输入序列长度
            - output_len: 输出序列长度
    
    Returns:
        (dataset, dataloader, node_locations)
    """
    from odps import ODPS
    
    # 获取 ODPS 凭证
    access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    
    assert access_id and secret, "缺少 ODPS 凭证环境变量"
    
    # 连接 ODPS
    odps = ODPS(
        access_id, 
        secret, 
        config['odps_project'],
        endpoint=config.get('odps_endpoint', 'http://service-corp.odps.aliyun-inc.com/api')
    )
    
    # 创建数据集
    dataset = PreprocessedODPSDataset(
        odps_instance=odps,
        table_name=config['odps_table'],
        meta_table_name=config['odps_meta_table'],
        adcode=config['adcode'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        input_len=config.get('input_len', 12),
        output_len=config.get('output_len', 12),
        log=log
    )
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=0,  # ODPS 数据已经在内存中，不需要多线程
        pin_memory=True
    )
    
    return dataset, dataloader, dataset.node_locations


if __name__ == '__main__':
    """测试预处理数据加载器"""
    
    config = {
        'odps_project': 'autonavi_traffic_report',
        'odps_endpoint': 'http://service-corp.odps.aliyun-inc.com/api',
        'odps_table': 'tb_inter_traffic_timeseries',
        'odps_meta_table': 'tb_inter_node_metadata',
        'adcode': '650100',
        'start_date': '20250919',
        'end_date': '20250925',
        'batch_size': 32,
        'num_workers': 0,
        'input_len': 12,
        'output_len': 12
    }
    
    print("=" * 80)
    print("测试预处理数据加载器")
    print("=" * 80)
    
    dataset, dataloader, locations = create_preprocessed_dataloader(config)
    
    print(f"\n数据集信息:")
    print(f"  - 节点数: {dataset.num_nodes}")
    print(f"  - 样本数: {len(dataset)}")
    print(f"  - Batch 数: {len(dataloader)}")
    
    print(f"\n位置信息:")
    print(f"  - Shape: {locations.shape}")
    print(f"  - 范围: [{locations.min():.4f}, {locations.max():.4f}]")
    
    print(f"\n加载前 3 个 batch:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        X = batch['X']  # (batch, T, N, 1)
        Y = batch['Y']  # (batch, T', N, 1)
        TE = batch['TE']  # (batch, T, 2)
        
        print(f"\nBatch {i+1}:")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  TE shape: {TE.shape}")
        
        # 验证数据
        print(f"  X 范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  Y 范围: [{Y.min():.2f}, {Y.max():.2f}]")
        print(f"  非零比例: {(X > 0).sum() / X.size * 100:.2f}%")
        
        # 验证所有节点都有数据（非稀疏）
        nodes_with_data = (X.sum(axis=(0, 1)) > 0).sum()
        print(f"  有数据的节点数: {nodes_with_data} / {dataset.num_nodes}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
