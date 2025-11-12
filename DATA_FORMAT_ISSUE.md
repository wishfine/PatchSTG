# 数据格式问题说明

## 🎯 核心问题

### PatchSTG 原始数据格式
```python
X: (T, N, 1)
Y: (T', N, 1)
TE: (T, 2)

其中:
- T: 输入时间步数（如 12）
- N: 节点总数（如 325）
- 1: 流量特征
- T': 输出时间步数（如 12）

# 含义：一个样本包含所有节点在连续时间段的数据
# 例如：2025-07-01 08:00 到 08:11 这 12 分钟，325 个路口的流量数据
```

### ODPS 表的数据格式
```python
每条记录:
- nds_id, next_nds_id: 一个转向流（节点）
- passts_time: 一个时刻
- flow_label: 这个节点在这个时刻的流量
- time_feat: 过去 24 分钟的时间特征（字符串）
- dym_feat_feat: 过去 24 分钟的历史流量（字符串）

# 含义：单个节点，单个时刻的记录
# 需要重组成 (T, N, 1) 格式
```

## 🔄 数据重组策略

### 问题
ODPS 表是**长格式**（每条记录是一个 (node, time) 对），需要转换成**宽格式**（一个样本包含所有节点的时间序列）。

### 方案 A：在 ODPS SQL 中重组（推荐）

**优点**：
- 数据量小，传输快
- 逻辑清晰
- 容易调试

**实现**：
```sql
-- 创建预处理表：按时间窗口聚合所有节点
CREATE TABLE preprocessed_data AS
SELECT 
    time_window,  -- 时间窗口ID（如 2025-07-01 08:00）
    nds_id,
    next_nds_id,
    flow_sequence,  -- 该节点在这个窗口的流量序列（12个值）
    time_features   -- 时间特征
FROM (
    -- 使用窗口函数生成时间序列
    SELECT 
        DATE_FORMAT(passts_time, 'yyyy-MM-dd HH:mm') as time_window,
        nds_id,
        next_nds_id,
        WM_CONCAT(',', CAST(flow_label AS STRING)) OVER (
            PARTITION BY nds_id, next_nds_id
            ORDER BY passts_time
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) as flow_sequence
    FROM tb_inter_spatial_method_pretrain_data
    WHERE ...
)
```

然后每条记录变成：
```
time_window='2025-07-01 08:11', nds_id=xxx, next_nds_id=yyy, flow_sequence='5,3,2,1,0,0,8,15,12,10,8,6'
```

### 方案 B：在 Python 中重组（当前实现）

**问题**：
- ODPS表记录是按 `(node, time)` 组织的
- 每个 batch 可能只包含部分节点的部分时间
- 无法构建完整的 (T, N, 1) 矩阵

**当前的错误做法**：
```python
# 第一个 batch 确定节点列表（假设 50 个节点）
node_list = [...50 nodes...]

# 后续每个 batch
X = np.zeros((batch_size, T, 50, 1))  # ❌ 问题！
# 每条记录只填充一个节点：
X[sample_i, :, node_idx, 0] = [...]

# 结果：X 是稀疏的，每个样本只有1个节点有值
```

**正确做法**：
```python
# 需要按时间窗口组织数据
samples_by_time = {}  # {time_window: {node_id: flow_sequence}}

for record in all_records:
    time_window = record['passts_time']
    node_id = (record['nds_id'], record['next_nds_id'])
    flow_seq = parse(record['dym_feat_feat'])[:12]
    
    if time_window not in samples_by_time:
        samples_by_time[time_window] = {}
    samples_by_time[time_window][node_id] = flow_seq

# 然后构建 (T, N, 1) 样本
for time_window, node_data in samples_by_time.items():
    sample_X = np.zeros((T, N, 1))
    for node_idx, node_id in enumerate(node_list):
        if node_id in node_data:
            sample_X[:, node_idx, 0] = node_data[node_id]
```

### 方案 C：修改 ODPS 表结构（最优）

**创建符合 PatchSTG 格式的表**：

```sql
-- 表结构
CREATE TABLE tb_inter_traffic_matrix (
    time_window STRING,      -- '2025-07-01 08:11'
    adcode STRING,           -- '650100'
    flow_matrix STRING,      -- N 个节点的流量，逗号分隔
    time_features STRING,    -- 时间特征
    node_list STRING         -- 节点列表（元数据）
)
PARTITIONED BY (ds STRING);

-- 数据示例
time_window='2025-07-01 08:11'
adcode='650100'
flow_matrix='5,3,0,15,8,12,...' (12392 个值，对应 12392 个节点)
time_features='5 17 36 0 18 8'
```

这样每条记录就是一个完整的样本！

## 🚀 推荐方案

### 短期方案（方案 B 改进版）

修改 `ODPSTableDataLoader`，在加载时按时间窗口重组：

```python
def _load_and_reorganize(self):
    """按时间窗口重组数据"""
    
    # 1. 扫描所有记录，按时间窗口分组
    time_windows = {}  # {time_window: {node_id: flow_data}}
    
    for record in all_records:
        time_window = self._get_time_window(record['passts_time'])
        node_id = (record['nds_id'], record['next_nds_id'])
        
        if time_window not in time_windows:
            time_windows[time_window] = {}
        
        # 解析历史流量
        flow_seq = self._parse_flow(record['dym_feat_feat'])[:12]
        time_windows[time_window][node_id] = flow_seq
    
    # 2. 为每个时间窗口构建样本
    samples = []
    for time_window in sorted(time_windows.keys()):
        node_data = time_windows[time_window]
        
        # 构建 (T, N, 1) 矩阵
        X = np.zeros((12, len(self.node_list), 1))
        for node_idx, node_id in enumerate(self.node_list):
            if node_id in node_data:
                X[:, node_idx, 0] = node_data[node_id]
        
        samples.append(X)
    
    return samples
```

### 长期方案（方案 C）

创建预处理 ODPS 表，每条记录直接是 (T, N, 1) 格式的序列化数据。

## 📊 数据流对比

### 原始 NPZ 文件
```
文件: CA.npz
内容: 
  - X: (num_samples, T, N, 1) - 所有样本已经组织好
  - TE: (num_samples, T, 2) - 时间特征
```

### ODPS 原始表（当前）
```
每条记录: (node, time, flow)
需要: 重组成 (time_window, all_nodes, flows)
```

### ODPS 预处理表（建议）
```
每条记录: (time_window, flow_matrix)
flow_matrix 已经包含所有节点的流量
```

## ⚠️ 当前实现的问题

当前的 `ODPSTableDataLoader._collate_fn` 实现：
```python
# 问题 1: 每个样本只有一个节点有值
X[sample_i, :, node_idx, 0] = flow_sequence  # ❌ 其他节点都是 0

# 问题 2: batch_size 的含义错误
# 应该是: batch_size 个时间窗口（每个包含所有节点）
# 实际是: batch_size 条 ODPS 记录（每条只有一个节点）
```

正确的应该是：
```python
# 每个样本包含所有节点在同一时间窗口的数据
X[sample_i, :, :, 0] = all_nodes_flow_matrix  # (T, N)
```

## 🔧 修复方案

两个选择：

### 选择 1: 修改数据加载逻辑（复杂）
- 需要大量内存（存储所有记录）
- 需要复杂的重组逻辑
- 效率较低

### 选择 2: 预处理 ODPS 表（推荐）
- 一次性重组数据
- 训练时直接读取
- 效率高，逻辑简单

**建议**：
1. 先创建一个小的预处理表（1天数据）
2. 测试训练流程
3. 确认无误后处理全部数据

## 💡 下一步行动

1. **确认数据组织方式**：你们是否可以创建预处理表？
2. **临时方案**：使用方案 B 修改加载逻辑（内存占用大）
3. **长期方案**：使用方案 C 预处理数据（推荐）
