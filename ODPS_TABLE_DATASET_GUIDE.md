# 使用 OdpsTableDataset 直接读取 ODPS 表的完整指南

## 🎯 目标

将你的 ODPS 表数据用 `OdpsTableDataset` 方式读取，而不是用 SQL 查询的方式。

## 📊 你的数据表结构

### 主表: `autonavi_traffic_report.tb_inter_spatial_method_pretrain_data`

| 字段 | 类型 | 说明 |
|------|------|------|
| `nds_id` | bigint | 转向前的路段ID |
| `next_nds_id` | bigint | 转向后的路段ID |
| `adcode` | bigint | 城市代码 |
| `ds` | string | 日期分区 (YYYYMMDD) |
| `passts_time` | string | 时间戳 |
| `flow_label` | bigint | **预测目标**：当前时刻流量 |
| `time_feat` | string | **时间特征序列**：24个时间步的上下文 |
| `dym_feat_feat` | string | **历史流量序列**：过去24分钟的流量 |

### 元数据表: `autonavi_traffic_report.intersection_meta_1`

| 字段 | 类型 | 说明 |
|------|------|------|
| `inter_id` | string | 路口ID |
| `nds_id` | bigint | 转向前路段 |
| `next_nds_id` | bigint | 转向后路段 |
| `lat` | double | 路口纬度 |
| `lng` | double | 路口经度 |
| `adcode` | string | 城市代码 |

## 🔧 实现方案

### 方案 1：直接使用主表（推荐先测试）

**优点**：
- ✅ 数据已经是字符串格式，可以直接用 `OdpsTableDataset` 读取
- ✅ 不需要 SQL JOIN，读取速度快
- ✅ 适合大规模训练

**实现步骤**：

#### 1. 创建 `collate_fn`

```python
import numpy as np
import torch

def collate_fn_patchstg(batch):
    """
    将 ODPS 表记录转换为 PatchSTG 模型需要的格式
    
    输入 batch: 列表，每个元素是一条记录
        [nds_id, next_nds_id, adcode, ds, passts_time, flow_label, time_feat, dym_feat_feat]
    
    输出:
        - node_ids: (batch_size, 2) - 转向流ID对
        - time_features: (batch_size, 24, 6) - 时间特征
        - flow_features: (batch_size, 24, 1) - 历史流量
        - labels: (batch_size, 1) - 预测目标
    """
    
    node_ids = []
    time_feats = []
    flow_feats = []
    labels = []
    
    for record in batch:
        nds_id = record[0]
        next_nds_id = record[1]
        flow_label = record[5]
        time_feat_str = record[6]
        dym_feat_str = record[7]
        
        # 解析 time_feat: "5 17 36 0 18 8;5 17 35 0 18 8;..."
        time_array = np.array([
            [int(x) for x in seg.split(' ')]
            for seg in time_feat_str.split(';')
        ], dtype=np.int32)  # Shape: (24, 6)
        
        # 解析 dym_feat_feat: "0;0;2;1;1;..."
        flow_array = np.array([
            [float(x)] for x in dym_feat_str.split(';')
        ], dtype=np.float32)  # Shape: (24, 1)
        
        node_ids.append([nds_id, next_nds_id])
        time_feats.append(time_array)
        flow_feats.append(flow_array)
        labels.append([float(flow_label)])
    
    return {
        'node_ids': torch.tensor(node_ids, dtype=torch.long),
        'time_features': torch.from_numpy(np.stack(time_feats)),
        'flow_features': torch.from_numpy(np.stack(flow_feats)),
        'labels': torch.tensor(labels, dtype=torch.float32)
    }
```

#### 2. 创建 DataLoader

```python
from utils.odps_table import OdpsTableDataset
from torch.utils.data import DataLoader

# 表路径（ODPS 格式）
odps_table = "odps://autonavi_traffic_report/tables/tb_inter_spatial_method_pretrain_data"

# 创建数据集
dataset = OdpsTableDataset(
    odps_table, 
    slice_id=0,      # 当前进程ID（分布式训练时使用）
    slice_count=1    # 总进程数
)

# 创建 DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_patchstg,
    pin_memory=True,
    prefetch_factor=8
)

# 使用
for batch in data_loader:
    node_ids = batch['node_ids']          # (batch, 2)
    time_features = batch['time_features']  # (batch, 24, 6)
    flow_features = batch['flow_features']  # (batch, 24, 1)
    labels = batch['labels']              # (batch, 1)
    
    # 训练模型...
```

### 方案 2：关联元数据表获取经纬度

**问题**：`OdpsTableDataset` 通常只读单表，如果要 JOIN 需要：

#### 选项 A：预先创建视图或新表

```sql
-- 创建包含经纬度的完整表
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_with_location AS
SELECT 
    f.nds_id,
    f.next_nds_id,
    f.adcode,
    f.ds,
    f.passts_time,
    f.flow_label,
    f.time_feat,
    f.dym_feat_feat,
    m.inter_id,
    m.lat,
    m.lng
FROM autonavi_traffic_report.tb_inter_spatial_method_pretrain_data f
LEFT JOIN autonavi_traffic_report.intersection_meta_1 m
    ON f.nds_id = m.nds_id 
    AND f.next_nds_id = m.next_nds_id
    AND f.adcode = CAST(m.adcode AS BIGINT)
WHERE f.ds >= '20250901' AND f.ds <= '20251031';
```

然后直接读这个新表。

#### 选项 B：分别读取并在内存中关联

```python
# 1. 先读元数据表，构建位置字典
def load_location_dict():
    from odps import ODPS
    import os
    
    odps = ODPS(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        project='autonavi_traffic_report',
        endpoint='http://service-corp.odps.aliyun-inc.com/api'
    )
    
    query = """
    SELECT nds_id, next_nds_id, inter_id, lat, lng
    FROM autonavi_traffic_report.intersection_meta_1
    WHERE adcode = '650100'
    """
    
    location_dict = {}
    with odps.execute_sql(query).open_reader() as reader:
        for record in reader:
            key = (record[0], record[1])  # (nds_id, next_nds_id)
            location_dict[key] = {
                'inter_id': record[2],
                'lat': record[3],
                'lng': record[4]
            }
    
    return location_dict

# 2. 在 collate_fn 中使用
location_dict = load_location_dict()

def collate_fn_with_location(batch):
    # ... 前面的解析逻辑 ...
    
    locations = []
    for record in batch:
        nds_id = record[0]
        next_nds_id = record[1]
        key = (nds_id, next_nds_id)
        
        if key in location_dict:
            loc = location_dict[key]
            locations.append([loc['lat'], loc['lng']])
        else:
            locations.append([0.0, 0.0])  # 缺失值填充
    
    return {
        # ... 其他字段 ...
        'locations': torch.tensor(locations, dtype=torch.float32)  # (batch, 2)
    }
```

## 📝 完整示例代码

```python
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.odps_table import OdpsTableDataset

# ========== 配置 ==========
ODPS_PROJECT = 'autonavi_traffic_report'
ODPS_TABLE = 'tb_inter_spatial_method_pretrain_data'
ADCODE = '650100'  # 乌鲁木齐

# ========== collate_fn ==========
def collate_fn(batch):
    """数据批处理函数"""
    
    node_ids = []
    time_feats = []
    flow_feats = []
    labels = []
    
    for record in batch:
        # 字段索引
        nds_id = record[0]
        next_nds_id = record[1]
        flow_label = record[5]
        time_feat_str = record[6]
        dym_feat_str = record[7]
        
        # 解析时间特征: "5 17 36 0 18 8;..."
        try:
            time_array = np.array([
                [int(x) for x in seg.split(' ')]
                for seg in time_feat_str.split(';')
            ], dtype=np.int32)
        except:
            time_array = np.zeros((24, 6), dtype=np.int32)
        
        # 解析流量特征: "0;0;2;1;..."
        try:
            flow_array = np.array([
                float(x) for x in dym_feat_str.split(';')
            ], dtype=np.float32)
        except:
            flow_array = np.zeros(24, dtype=np.float32)
        
        node_ids.append([nds_id, next_nds_id])
        time_feats.append(time_array)
        flow_feats.append(flow_array)
        labels.append(float(flow_label))
    
    return {
        'node_ids': torch.tensor(node_ids, dtype=torch.long),
        'time_features': torch.from_numpy(np.stack(time_feats)),
        'flow_features': torch.from_numpy(np.stack(flow_feats)),
        'labels': torch.tensor(labels, dtype=torch.float32)
    }

# ========== 创建 DataLoader ==========
def create_dataloader(batch_size=64, num_workers=4):
    
    odps_table_path = f"odps://{ODPS_PROJECT}/tables/{ODPS_TABLE}"
    
    dataset = OdpsTableDataset(
        odps_table_path,
        slice_id=0,
        slice_count=1
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=8
    )
    
    return data_loader

# ========== 测试 ==========
if __name__ == '__main__':
    loader = create_dataloader(batch_size=32)
    
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  node_ids: {batch['node_ids'].shape}")
        print(f"  time_features: {batch['time_features'].shape}")
        print(f"  flow_features: {batch['flow_features'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        
        if i == 0:
            break
```

## ⚠️ 注意事项

### 1. 表分区问题

如果表有分区（如 `ds=20250919`），需要指定：

```python
# 方式1: 在表名中指定分区
odps_table = "odps://autonavi_traffic_report/tables/tb_inter_spatial_method_pretrain_data/ds=20250919"

# 方式2: 如果 OdpsTableDataset 支持过滤，传入参数
dataset = OdpsTableDataset(
    odps_table,
    slice_id=0,
    slice_count=1,
    filters={'ds': '20250919', 'adcode': '650100'}  # 需要确认是否支持
)
```

### 2. 数据量控制

ODPS 表可能有数十亿条记录，建议：
- 只读特定日期范围的分区
- 使用分布式训练时，设置正确的 `slice_id` 和 `slice_count`

### 3. 时间特征对齐

确保 `time_feat` 中的 24 个时间步与 `dym_feat_feat` 对齐：
- `time_feat[0]` 对应 `dym_feat_feat[0]`（都是前1分钟）
- `time_feat[23]` 对应 `dym_feat_feat[23]`（都是前24分钟）

## 🚀 下一步

1. **先运行测试脚本** `check_odps_direct.py` 验证能否读取数据
2. **检查数据质量**：查看 `time_feat` 和 `dym_feat_feat` 是否有缺失或异常
3. **修改 PatchSTG 的 DataLoader**：将原来的 SQL 查询方式改为 `OdpsTableDataset`
4. **性能优化**：调整 `batch_size`、`num_workers`、`prefetch_factor` 等参数

## 📚 参考

- 原 notebook 中的 `collate_fn` 实现
- PatchSTG 的数据加载逻辑
- ODPS Table 分区读取文档
