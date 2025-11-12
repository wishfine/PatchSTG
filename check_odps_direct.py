"""
使用 OdpsTableDataset 直接读取 ODPS 表的方式
适配你的表结构: autonavi_traffic_report.tb_inter_spatial_method_pretrain_data
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys

# 假设你有 OdpsTableDataset 工具类，如果没有需要安装或实现
# 这里先写一个模拟版本，实际使用时替换成真实的
try:
    from utils.odps_table import OdpsTableDataset
except:
    print("⚠️  未找到 OdpsTableDataset，需要从原项目导入或实现")
    # 简单的模拟实现用于测试
    class OdpsTableDataset:
        def __init__(self, table_path, slice_id=0, slice_count=1):
            """模拟的 ODPS 表读取器"""
            from odps import ODPS
            
            access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
            secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
            
            # 解析 table_path: odps://project/tables/table_name
            parts = table_path.replace('odps://', '').split('/tables/')
            project = parts[0]
            table_name = parts[1]
            
            self.odps = ODPS(
                access_id, 
                secret, 
                project=project,
                endpoint='http://service-corp.odps.aliyun-inc.com/api'
            )
            self.table_name = table_name
            self.slice_id = slice_id
            self.slice_count = slice_count
            self._data = None
            self._load_data()
        
        def _load_data(self):
            """加载数据"""
            # 简单查询，实际应该支持分片
            query = f"""
            SELECT nds_id, next_nds_id, adcode, ds, passts_time, 
                   flow_label, time_feat, dym_feat_feat
            FROM {self.table_name}
            WHERE adcode = '650100' 
              AND ds >= '20250919' 
              AND ds <= '20250925'
            LIMIT 100
            """
            print(f"Executing query: {query}")
            
            with self.odps.execute_sql(query).open_reader() as reader:
                self._data = [list(record.values) for record in reader]
            
            print(f"Loaded {len(self._data)} records")
        
        def __len__(self):
            return len(self._data) if self._data else 0
        
        def __getitem__(self, idx):
            """返回一条记录"""
            return self._data[idx]


# ========== 数据列定义 ==========
data_columns = [
    'nds_id',           # bigint
    'next_nds_id',      # bigint  
    'adcode',           # bigint
    'ds',               # string
    'passts_time',      # string
    'flow_label',       # bigint - 预测目标
    'time_feat',        # string - 时间特征序列
    'dym_feat_feat'     # string - 历史流量序列
]

# 列索引映射
COL_NDS_ID = 0
COL_NEXT_NDS_ID = 1
COL_ADCODE = 2
COL_DS = 3
COL_PASSTS_TIME = 4
COL_FLOW_LABEL = 5
COL_TIME_FEAT = 6
COL_DYM_FEAT = 7


def collate_fn(batch):
    """
    批量数据处理函数
    
    输入: batch 是一个列表，每个元素是一条记录 [nds_id, next_nds_id, ..., time_feat, dym_feat_feat]
    输出: 
        - nds_ids: (batch_size,)
        - next_nds_ids: (batch_size,)
        - time_features: (batch_size, 24, 6) - 24个时间步，每步6个特征
        - flow_features: (batch_size, 24) - 24个历史流量值
        - labels: (batch_size,) - 当前时刻流量标签
        - adcodes: (batch_size,)
    """
    
    nds_id_list = []
    next_nds_id_list = []
    adcode_list = []
    time_feat_list = []
    flow_feat_list = []
    label_list = []
    
    for record in batch:
        # 提取各字段
        nds_id = record[COL_NDS_ID]
        next_nds_id = record[COL_NEXT_NDS_ID]
        adcode = record[COL_ADCODE]
        flow_label = record[COL_FLOW_LABEL]
        time_feat_str = record[COL_TIME_FEAT]
        dym_feat_str = record[COL_DYM_FEAT]
        
        # 解析 time_feat: "5 17 36 0 18 8;5 17 35 0 18 8;..."
        # 每段包含6个数字: [week, hour, minute, day_type, day, month]
        try:
            time_feat_array = np.array([
                [int(x) for x in segment.split(' ')]
                for segment in time_feat_str.split(';')
            ], dtype=np.int32)  # Shape: (24, 6)
        except:
            # 如果解析失败，用零填充
            time_feat_array = np.zeros((24, 6), dtype=np.int32)
        
        # 解析 dym_feat_feat: "0;0;2;1;1;0;..."
        # 每段是一个流量值
        try:
            flow_feat_array = np.array([
                float(x) for x in dym_feat_str.split(';')
            ], dtype=np.float32)  # Shape: (24,)
        except:
            flow_feat_array = np.zeros(24, dtype=np.float32)
        
        # 添加到列表
        nds_id_list.append(nds_id)
        next_nds_id_list.append(next_nds_id)
        adcode_list.append(adcode)
        time_feat_list.append(time_feat_array)
        flow_feat_list.append(flow_feat_array)
        label_list.append(float(flow_label))
    
    # 转换为 tensor
    nds_ids = torch.tensor(nds_id_list, dtype=torch.long)
    next_nds_ids = torch.tensor(next_nds_id_list, dtype=torch.long)
    adcodes = torch.tensor(adcode_list, dtype=torch.long)
    time_features = torch.from_numpy(np.stack(time_feat_list))  # (batch, 24, 6)
    flow_features = torch.from_numpy(np.stack(flow_feat_list))  # (batch, 24)
    labels = torch.tensor(label_list, dtype=torch.float32)
    
    return {
        'nds_id': nds_ids,
        'next_nds_id': next_nds_ids,
        'adcode': adcodes,
        'time_features': time_features,
        'flow_features': flow_features,
        'labels': labels
    }


def test_direct_read():
    """测试直接读取 ODPS 表"""
    
    print("=" * 60)
    print("测试使用 OdpsTableDataset 直接读取 ODPS 表")
    print("=" * 60)
    
    # 表路径（ODPS格式）
    odps_table_path = "odps://autonavi_traffic_report/tables/tb_inter_spatial_method_pretrain_data"
    
    print(f"\n📊 表路径: {odps_table_path}")
    
    # 创建数据集
    slice_id = 0  # 当前分片ID（分布式训练时使用）
    slice_count = 1  # 总分片数
    
    try:
        dataset = OdpsTableDataset(odps_table_path, slice_id, slice_count)
        print(f"✅ 数据集创建成功，共 {len(dataset)} 条记录")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return
    
    # 创建 DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"\n📦 DataLoader 创建成功，batch_size=4")
    
    # 测试读取一个 batch
    print("\n" + "=" * 60)
    print("测试读取第一个 batch")
    print("=" * 60)
    
    for batch_idx, batch in enumerate(data_loader):
        print(f"\n🔹 Batch {batch_idx + 1}:")
        print(f"  nds_id shape: {batch['nds_id'].shape}")
        print(f"  nds_id values: {batch['nds_id']}")
        print(f"  next_nds_id shape: {batch['next_nds_id'].shape}")
        print(f"  adcode shape: {batch['adcode'].shape}")
        print(f"  adcode values: {batch['adcode']}")
        print(f"  time_features shape: {batch['time_features'].shape}")  # (batch, 24, 6)
        print(f"  time_features[0, 0, :]: {batch['time_features'][0, 0, :]}")  # 第一个样本第一个时间步
        print(f"  flow_features shape: {batch['flow_features'].shape}")  # (batch, 24)
        print(f"  flow_features[0, :]: {batch['flow_features'][0, :]}")  # 第一个样本的历史流量
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  labels values: {batch['labels']}")
        
        # 只读取第一个 batch
        break
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


def test_with_meta_table():
    """测试关联元数据表获取经纬度"""
    
    print("\n" + "=" * 60)
    print("测试关联元数据表获取路口经纬度")
    print("=" * 60)
    
    from odps import ODPS
    
    access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    
    odps = ODPS(
        access_id, 
        secret, 
        project='autonavi_traffic_report',
        endpoint='http://service-corp.odps.aliyun-inc.com/api'
    )
    
    # 查询：关联主表和元数据表
    query = """
    SELECT 
        f.nds_id,
        f.next_nds_id,
        f.adcode,
        f.flow_label,
        m.inter_id,
        m.lat,
        m.lng
    FROM autonavi_traffic_report.tb_inter_spatial_method_pretrain_data f
    LEFT JOIN autonavi_traffic_report.intersection_meta_1 m
        ON f.nds_id = m.nds_id 
        AND f.next_nds_id = m.next_nds_id
        AND f.adcode = m.adcode
    WHERE f.adcode = '650100' 
      AND f.ds = '20250919'
    LIMIT 10
    """
    
    print(f"\n执行关联查询:")
    print(query)
    
    try:
        with odps.execute_sql(query).open_reader() as reader:
            results = []
            for record in reader:
                results.append(record.values)
            
            print(f"\n✅ 查询成功，返回 {len(results)} 条记录\n")
            
            # 显示结果
            for i, row in enumerate(results[:5]):
                nds_id, next_nds_id, adcode, flow_label, inter_id, lat, lng = row
                print(f"记录 {i+1}:")
                print(f"  转向流: ({nds_id}, {next_nds_id})")
                print(f"  路口: {inter_id}")
                print(f"  位置: ({lat}, {lng})")
                print(f"  流量: {flow_label}")
                print()
            
            # 统计有多少记录有经纬度
            has_location = sum(1 for row in results if row[5] is not None)
            print(f"📊 统计:")
            print(f"  总记录数: {len(results)}")
            print(f"  有经纬度: {has_location}")
            print(f"  覆盖率: {has_location/len(results)*100:.1f}%")
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")


if __name__ == '__main__':
    # 测试1: 直接读取主表
    test_direct_read()
    
    # 测试2: 关联元数据表
    test_with_meta_table()
