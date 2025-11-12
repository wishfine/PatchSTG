# 使用 ODPS 数据训练 PatchSTG 模型

## 概述

本文档说明如何使用 MaxCompute (ODPS) 中的交通流量数据表 `autonavi_traffic_report.tb_inter_spatial_method_pretrain_data` 来训练 PatchSTG 模型。

## 数据表结构

### 表名
`autonavi_traffic_report.tb_inter_spatial_method_pretrain_data`

## 元数据表（可选但推荐）

### 为什么需要元数据表？

PatchSTG 的核心特性是使用 **KD-tree 进行空间 patching**，将地理上相近的路段划分到同一个 patch 中。这需要路段的地理位置信息（经纬度）。

**有元数据表**:
- ✅ 使用 KD-tree 进行真实的空间划分
- ✅ 保留空间局部性，提高模型性能
- ✅ 符合论文中的方法

**无元数据表**:
- ⚠️ 退化到简单的顺序 patching
- ⚠️ 所有节点放在一个 patch 中
- ⚠️ 可能降低模型性能

### 元数据表结构

**表名建议**: `autonavi_traffic_report.tb_inter_spatial_node_location`

**数据关系说明**：
- `(nds_id, next_nds_id)` 表示一个转向流（从一条路转向另一条路）
- 每个转向流对应一个路口 `inter_id`
- 每个路口有地理坐标（经纬度）
- 一个路口可能有多个转向流（例如十字路口有12种转向）

**必需字段**:

| 字段名 | 类型 | 说明 | 必需 |
|--------|------|------|------|
| `nds_id` | STRING | 转向前的路段 ID（入路段） | ✅ |
| `next_nds_id` | STRING | 转向后的路段 ID（出路段） | ✅ |
| `inter_id` | STRING | 路口 ID（转向发生的路口） | ✅ |
| `latitude` | DOUBLE | 路口纬度 | ✅ |
| `longitude` | DOUBLE | 路口经度 | ✅ |
| `adcode` | STRING | 行政区划代码 | 推荐 |

**可选字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `intersection_name` | STRING | 路口名称 |
| `intersection_type` | STRING | 路口类型（十字、T型等） |
| `turn_type` | STRING | 转向类型（左转、右转、直行等） |

> 📖 **详细的元数据表创建指南**: 请参阅 [META_TABLE_GUIDE.md](META_TABLE_GUIDE.md)

### 字段说明

**主数据表 (tb_inter_spatial_method_pretrain_data)**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `nds_id` | STRING/BIGINT | 转向前的路段 ID（入路段） |
| `next_nds_id` | STRING/BIGINT | 转向后的路段 ID（出路段） |
| `adcode` | STRING | 行政区划代码（如 110000=北京） |
| `ds` | STRING | 数据分区日期 (YYYYMMDD) |
| `passts_time` | DATETIME/STRING | 当前时刻（预测目标时刻） |
| `flow_label` | BIGINT | 目标变量：该时刻的实际车流量 |
| `time_feat` | STRING | 时间特征序列（24段，分号分隔）<br>格式: "week hour minute day_type day month;..." |
| `dym_feat_feat` | STRING | 动态流量特征序列（24个流量值，分号分隔）<br>格式: "15;8;0;12;...;3" |

**说明**：
- `(nds_id, next_nds_id)` 表示一个转向流：从 nds_id 路段转向 next_nds_id 路段
- 每个转向流在某个路口发生，这个路口有具体的地理坐标（在元数据表中）
- 不同转向流可能对应同一个路口（例如一个路口有多个转向方向）

**元数据表 (tb_inter_spatial_node_location，可选)**:

**关系说明**: 转向流 → 路口 → 坐标
- 每个 (nds_id, next_nds_id) 对应一个路口 inter_id
- 路口 inter_id 有具体的经纬度坐标

| 字段名 | 类型 | 说明 | 必需 |
|--------|------|------|------|
| `nds_id` | STRING | 转向前的路段 ID（入路段） | ✅ |
| `next_nds_id` | STRING | 转向后的路段 ID（出路段） | ✅ |
| `inter_id` | STRING | 路口 ID | ✅ |
| `latitude` | DOUBLE | 路口纬度 | ✅ |
| `longitude` | DOUBLE | 路口经度 | ✅ |
| `adcode` | STRING | 行政区划代码 | 推荐 |

> 📖 **详细的元数据表创建指南**: 请参阅 [META_TABLE_GUIDE.md](META_TABLE_GUIDE.md)

### 时间特征 (time_feat) 详解

每段包含 6 个维度（空格分隔）：
- `time_week`: 星期几 (0=周一, ..., 6=周日)
- `time_hour`: 小时 (0~23)
- `time_minute`: 分钟 (0~59)
- `day_type`: 日期类型 (0=工作日, 1=周末, 2=节假日)
- `time_day`: 日 (0~30，1号→0, 2号→1, ...)
- `time_month`: 月 (0~11，1月→0, ..., 12月→11)

## 环境配置

### 1. 设置 ODPS 凭证

在 `~/.zshrc` 或 `~/.bashrc` 中添加：

```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_access_key_id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_access_key_secret"
```

然后重新加载配置：

```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

### 2. 安装依赖

确保已安装以下 Python 包：

```bash
pip install pyodps pandas numpy torch tqdm
```

## 使用方法

### 方式 1: 使用配置文件训练

1. **编辑配置文件** `config/ODPS.conf`：

```ini
[data]
# ODPS 配置
odps_project = autonavi_traffic_report
odps_endpoint = http://service-corp.odps.aliyun-inc.com/api
odps_table = tb_inter_spatial_method_pretrain_data

# 元数据表（可选，用于空间 patching）
# 如果提供，将使用 KD-tree 进行真实的空间划分
# 如果不提供或留空，将使用简单的顺序 patching
odps_meta_table = tb_inter_spatial_node_location

# 行政区划代码
adcode = 110000  # 110000=北京, 310000=上海, 440100=广州, 440300=深圳

# 数据日期范围
start_date = 20250701
end_date = 20250731

# 序列长度
input_len = 12   # 输入 12 个时间步
output_len = 12  # 预测 12 个时间步
```

2. **运行训练脚本**：

```bash
# 训练模式
python train_odps.py --config config/ODPS.conf --mode train

# 测试模式
python train_odps.py --config config/ODPS.conf --mode test

# 训练+测试
python train_odps.py --config config/ODPS.conf --mode both
```

### 方式 2: 使用 Python 代码

```python
from lib.odps_data_loader import ODPSDataLoader
from train_odps import ODPSSolver

# 配置参数
config = {
    # ODPS 配置
    'odps_project': 'autonavi_traffic_report',
    'odps_endpoint': 'http://service-corp.odps.aliyun-inc.com/api',
    'odps_table': 'tb_inter_spatial_method_pretrain_data',
    
    # 数据过滤
    'adcode': '110000',  # 北京
    'start_date': '20250701',
    'end_date': '20250731',
    
    # 训练参数
    'input_len': 12,
    'output_len': 12,
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,
    'batch_size': 32,
    'max_epoch': 50,
    'learning_rate': 0.001,
    
    # 模型参数
    'layers': 3,
    'tem_patchsize': 4,
    'tem_patchnum': 3,
    'factors': 5,
    'spa_patchsize': 4,
    'spa_patchnum': 1,
    'tod': 1,
    'dow': 1,
    'input_dims': 1,
    'node_dims': 32,
    'tod_dims': 32,
    'dow_dims': 32,
    
    # 文件路径
    'model_file': './saved_models/odps_model.pth',
    'log_file': './log/odps_log',
    'cuda': '0',
    'seed': 1234,
}

# 打开日志
log = open(config['log_file'], 'w')

# 创建 solver 并训练
solver = ODPSSolver(config)
solver.train()
solver.test()

log.close()
```

### 方式 3: 分步加载和训练

```python
from lib.odps_data_loader import ODPSDataLoader
from train_odps import ODPSSolver

# 步骤 1: 先加载数据（可以单独运行，检查数据质量）
data_loader = ODPSDataLoader(config, log)
data_loader.load_data()

# 查看数据信息
info = data_loader.get_data_info()
print(f"训练样本: {info['train_samples']}")
print(f"验证样本: {info['val_samples']}")
print(f"测试样本: {info['test_samples']}")
print(f"节点数: {info['num_nodes']}")

# 步骤 2: 使用预加载的数据训练
solver = ODPSSolver(config, data_loader=data_loader)
solver.train()
solver.test()
```

## 数据加载流程

### ODPSDataLoader 的工作流程

1. **连接 ODPS**
   - 使用环境变量中的凭证初始化 ODPS 客户端

2. **执行 SQL 查询**
   ```sql
   SELECT nds_id, next_nds_id, adcode, ds, passts_time, 
          flow_label, time_feat, dym_feat_feat
   FROM tb_inter_spatial_method_pretrain_data
   WHERE adcode = '110000' 
     AND ds >= '20250701' 
     AND ds <= '20250731'
   ORDER BY nds_id, next_nds_id, passts_time
   ```

3. **构建节点列表**
   - 提取唯一的 `(nds_id, next_nds_id)` 对
   - 每对代表一个转向流（从一条路段转向另一条路段的交通流）
   - 如果提供元数据表，每个转向流会对应一个路口的地理坐标

4. **解析特征**
   - 解析 `time_feat`: 提取当前时刻的 hour 和 week 作为 tod, dow
   - 解析 `dym_feat_feat`: 提取历史流量序列（暂未使用）

5. **生成样本**
   - 使用滑动窗口方法
   - 输入: 过去 `input_len` 个时间步的流量
   - 输出: 未来 `output_len` 个时间步的流量

6. **划分数据集**
   - 按照 train_ratio, val_ratio, test_ratio 划分
   - 计算训练集的 mean 和 std 用于归一化

## 数据形状说明

### 输入数据 (X)
- **形状**: `(num_samples, input_len, num_nodes, 1)`
- **含义**: 
  - `num_samples`: 样本数量
  - `input_len`: 输入时间步长（如 12）
  - `num_nodes`: 路段数量（节点对数量）
  - `1`: 特征维度（流量值）

### 输出数据 (Y)
- **形状**: `(num_samples, output_len, num_nodes, 1)`
- **含义**: 未来时间步的流量预测目标

### 时间特征 (TE)
- **形状**: `(num_samples, time_len, 2)`
- **含义**: 
  - `time_len`: 时间步长（input_len 或 output_len）
  - `2`: [tod, dow] - 时刻和星期

## 配置参数说明

### ODPS 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `odps_project` | ODPS 项目名 | `autonavi_traffic_report` |
| `odps_endpoint` | ODPS 端点 | `http://service-corp.odps.aliyun-inc.com/api` |
| `odps_table` | 表名 | `tb_inter_spatial_method_pretrain_data` |

### 数据过滤参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `adcode` | 行政区划代码 | `110000` (北京) |
| `start_date` | 开始日期 | `20250701` |
| `end_date` | 结束日期 | `20250731` |

### 序列参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `input_len` | 输入序列长度 | 12 (12分钟) |
| `output_len` | 输出序列长度 | 12 (预测12分钟) |

### 数据集划分

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `train_ratio` | 训练集比例 | 0.7 |
| `val_ratio` | 验证集比例 | 0.1 |
| `test_ratio` | 测试集比例 | 0.2 |

## 城市代码参考

| 城市 | adcode |
|------|--------|
| 北京 | 110000 |
| 上海 | 310000 |
| 广州 | 440100 |
| 深圳 | 440300 |
| 杭州 | 330100 |
| 成都 | 510100 |

## 注意事项

### 1. 数据量控制

ODPS 查询可能返回大量数据，建议：
- 限制日期范围（如一个月）
- 选择特定城市（使用 adcode）
- 监控内存使用

### 节点数量

- 节点数 = 唯一的 `(nds_id, next_nds_id)` 对数量
- 每个对代表一个转向流（从一条路转向另一条路）
- 不同城市/时间范围的节点数不同
- 节点数会自动从数据中获取，无需手动配置
- 如果提供元数据表，多个转向流可能对应同一个路口（共享地理坐标）

### 3. 时间序列连续性

- 数据按 `passts_time` 排序
- 确保同一路段的数据时间连续
- 缺失时间点会影响样本生成

### 4. 归一化

- 使用训练集的 mean 和 std
- 验证集和测试集使用相同的归一化参数

## 示例输出

```
------------ Loading Data from ODPS -------------
Project: autonavi_traffic_report
Table: tb_inter_spatial_method_pretrain_data
Adcode: 110000
Date range: 20250701 ~ 20250731

Executing query:
SELECT nds_id, next_nds_id, adcode, ds, passts_time, 
       flow_label, time_feat, dym_feat_feat
FROM tb_inter_spatial_method_pretrain_data
WHERE adcode = '110000' 
  AND ds >= '20250701' 
  AND ds <= '20250731'
ORDER BY nds_id, next_nds_id, passts_time

Loaded 1234567 records from ODPS
Found 850 unique node pairs (road segments)
Generated 123000 samples in total

Train samples: 86100
Val samples: 12300
Test samples: 24600
Nodes: 850
Mean: 8.5432, Std: 12.3456
------------ End -------------
```

## 故障排除

### 问题 1: 凭证错误

**错误**: `缺少 ODPS 凭证`

**解决**:
```bash
# 检查环境变量
echo $ALIBABA_CLOUD_ACCESS_KEY_ID
echo $ALIBABA_CLOUD_ACCESS_KEY_SECRET

# 如果为空，设置凭证
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_key_id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_secret"
```

### 问题 2: 无数据返回

**错误**: `No data loaded from ODPS`

**解决**:
- 检查 adcode 是否正确
- 检查日期范围是否有数据
- 直接在 ODPS 控制台执行查询验证

### 问题 3: 内存不足

**错误**: `MemoryError`

**解决**:
- 缩短日期范围
- 减小 batch_size
- 使用更少的节点（选择特定区域）

### 问题 4: 节点数为 0

**错误**: `node_num = 0`

**解决**:
- 检查配置文件中的 nodes 参数
- 节点数会自动从数据获取，确保 `nodes = 0` 以使用自动检测

## 与原 NPZ 数据的区别

| 特性 | NPZ 数据 | ODPS 数据 |
|------|----------|-----------|
| 数据源 | 本地文件 | 远程数据库 |
| 节点定义 | 单个交叉口 | 转向流 (nds_id → next_nds_id) |
| 空间关系 | 固定邻接矩阵 | 基于路口坐标动态构建（可选） |
| 时间特征 | 预计算 | 实时解析 |
| 数据更新 | 手动 | 自动（查询最新数据） |
| 空间 patching | 基于节点坐标 | 基于路口坐标（转向流→路口→坐标） |

## 下一步

1. **数据探索**: 运行 `ODPSDataLoader` 并检查 `get_data_info()`
2. **小规模测试**: 使用短日期范围（如 3 天）进行快速测试
3. **调整参数**: 根据结果调整模型和训练参数
4. **扩展到其他城市**: 修改 adcode 在不同城市训练

## 参考文档

- [PyODPS 文档](https://pyodps.readthedocs.io/)
- [PatchSTG 原始论文](相关链接)
- [数据加载器使用说明](DATA_LOADER_README.md)
