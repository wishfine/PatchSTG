# 🔄 PatchSTG ODPS 数据训练完整流程

从 ODPS 表加载数据到模型训练的详细流程说明

---

## 📊 流程总览

```
[ODPS 表] 
    ↓
[1. 连接 ODPS] 
    ↓
[2. 执行 SQL 查询] 
    ↓
[3. 解析原始数据] 
    ↓
[4. 构建节点列表] 
    ↓
[5. 加载节点位置] 
    ↓
[6. 处理并划分数据] ← ⚠️ 当前问题所在
    ↓
[7. 创建空间 Patch]
    ↓
[8. 训练模型]
```

---

## 🚀 详细流程说明

### 阶段 0: 启动训练脚本

**文件**: `train_odps.py` (第 232-310 行)

```python
# 命令行启动
python train_odps.py --config config/ODPS.conf --mode train

# 流程:
1. 解析命令行参数 (argparse)
2. 读取配置文件 (configparser)
3. 设置随机种子
4. 创建 ODPSSolver 实例 → 触发数据加载
```

**关键代码**:
```python
solver = ODPSSolver(vars(args))  # 第 301 行
# ↓ 在 __init__ 中调用
self.data_loader = ODPSDataLoader(config, log)  # 第 38 行
self.data_loader.load_data()  # 第 39 行 → 触发数据加载流程
```

---

### 阶段 1: 连接 ODPS

**文件**: `lib/odps_data_loader.py` → `_init_odps_client()` (第 94-113 行)

```python
def _init_odps_client(self):
    # 1. 从环境变量获取凭证
    access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    
    # 2. 创建 ODPS 客户端
    self._odps_client = ODPS(
        access_id, secret,
        project=self.odps_project,  # 'autonavi_traffic_report'
        endpoint=self.odps_endpoint  # 'http://service-corp.odps.aliyun-inc.com/api'
    )
```

**输出**: ODPS 连接实例 `self._odps_client`

---

### 阶段 2: 构建并执行 SQL 查询

**文件**: `lib/odps_data_loader.py` → `_build_query()` + `load_data()` (第 115-206 行)

#### 2.1 构建查询语句

```python
def _build_query(self):
    query = f"""
    SELECT 
        nds_id,           -- 起点路段 ID
        next_nds_id,      -- 终点路段 ID
        adcode,           -- 城市代码
        ds,               -- 分区日期
        passts_time,      -- 通过时间（精确到秒）
        flow_label,       -- 当前流量值
        time_feat,        -- 时间特征（24段，分号分隔）
        dym_feat_feat     -- 历史流量（24个值，分号分隔）
    FROM {self.odps_table}
    WHERE adcode = '{self.adcode}'           -- 例如: '650100'
      AND ds >= '{self.start_date}'          -- 例如: '20250919'
      AND ds <= '{self.end_date}'            -- 例如: '20250925'
    ORDER BY nds_id, next_nds_id, passts_time
    """
```

**查询条件**:
- `adcode`: 过滤城市（乌鲁木齐 = '650100'）
- `ds`: 过滤日期范围（7天数据）
- `ORDER BY`: 按节点和时间排序

#### 2.2 执行查询并转为 DataFrame

```python
with self._odps_client.execute_sql(query).open_reader() as reader:
    records = [record.values for record in reader]
    df = pd.DataFrame(records, columns=[...])
```

**输出**: 
- DataFrame `df`，约 27,545,086 条记录
- 每条记录 = 一个转向流在一个时刻的数据

**数据示例**:
```
| nds_id  | next_nds_id | passts_time     | flow_label | time_feat              | dym_feat_feat          |
|---------|-------------|-----------------|------------|------------------------|------------------------|
| 123456  | 789012      | 2025-09-19 8:15 | 15         | "5 8 15 0 4 9;..."     | "5;3;2;1;0;0;8;15;..." |
| 123456  | 789012      | 2025-09-19 8:16 | 12         | "5 8 16 0 4 9;..."     | "15;5;3;2;1;0;0;8;..." |
| 234567  | 890123      | 2025-09-19 8:15 | 8          | "5 8 15 0 4 9;..."     | "3;2;1;0;0;8;15;12;..." |
```

---

### 阶段 3: 解析原始数据

**文件**: `lib/odps_data_loader.py` → `_parse_time_feat()` + `_parse_dym_feat()` (第 140-177 行)

#### 3.1 解析时间特征

```python
def _parse_time_feat(self, time_feat_str):
    """
    输入: "5 8 15 0 4 9;5 8 14 0 4 9;...;5 8 0 0 4 9"  # 24段
           ↑ ↑ ↑  ↑ ↑ ↑
           │ │ │  │ │ └─ month (月份)
           │ │ │  │ └─── day (日期类型)
           │ │ │  └───── day_type (是否周末)
           │ │ └──────── minute (分钟)
           │ └────────── hour (小时)
           └──────────── week (星期)
    
    输出: np.array, shape (24, 6)
    """
    segments = time_feat_str.split(';')
    features = []
    for seg in segments:
        parts = seg.strip().split()
        features.append([int(p) for p in parts])
    return np.array(features[:24])
```

#### 3.2 解析历史流量

```python
def _parse_dym_feat(self, dym_feat_str):
    """
    输入: "15;8;0;12;5;3;...;10"  # 24个值（过去24分钟的流量）
    
    输出: np.array, shape (24,)
    """
    values = dym_feat_str.split(';')
    features = [float(val.strip()) for val in values]
    return np.array(features[:24])
```

---

### 阶段 4: 构建节点列表

**文件**: `lib/odps_data_loader.py` → `_build_node_list()` (第 208-220 行)

```python
def _build_node_list(self, df):
    """
    从 DataFrame 中提取所有唯一的节点对
    
    输入: df (27,545,086 条记录)
    
    处理:
    1. 提取唯一的 (nds_id, next_nds_id) 对
    2. 为每个节点对分配索引 (0 到 N-1)
    3. 创建映射字典: {(nds_id, next_nds_id): node_idx}
    
    输出:
    - self.node_list: [(123456, 789012), (234567, 890123), ...]
    - self.node_num: 12392 (节点总数)
    - self.node_to_idx: {(123456, 789012): 0, (234567, 890123): 1, ...}
    """
    node_pairs = df[['nds_id', 'next_nds_id']].drop_duplicates()
    self.node_list = [(row['nds_id'], row['next_nds_id']) 
                      for _, row in node_pairs.iterrows()]
    self.node_num = len(self.node_list)
    self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
```

**输出**:
- `node_list`: 12,392 个唯一转向流
- `node_to_idx`: 节点到索引的映射

---

### 阶段 5: 加载节点位置（⚠️ 必须）

**文件**: `lib/odps_data_loader.py` → `_load_node_locations()` (第 222-312 行)

```python
def _load_node_locations(self):
    """
    从元数据表加载节点的经纬度信息
    
    ⚠️ 这一步是必须的！
    - PatchSTG 需要节点位置进行 KD-tree 空间分组
    - 如果缺少元数据表，初始化时会报错
    - 如果位置覆盖率 < 50%，也会报错
    
    SQL 查询:
    SELECT nds_id, next_nds_id, inter_id, lat, lng
    FROM intersection_meta_1
    WHERE adcode = '650100'
    
    处理:
    1. 为每个转向流查找对应的路口位置
    2. 创建位置矩阵 (2, N): [[lat1, lat2, ...], [lng1, lng2, ...]]
    3. 验证位置覆盖率（必须 >= 50%）
    
    输出:
    - self.node_locations: shape (2, 12392)
    """
```

**位置数据示例**:
```
node_locations:
  [[43.825, 43.830, 43.822, ...],  # 纬度
   [87.616, 87.620, 87.610, ...]]  # 经度
```

**质量检查**:
- ✅ 覆盖率 >= 50%：继续
- ❌ 覆盖率 < 50%：报错退出
- ⚠️ 如果完全没有元数据表：初始化时报错

---

### ⚠️ 阶段 6: 处理并划分数据（当前问题所在）

**文件**: `lib/odps_data_loader.py` → `_process_and_split_data()` (第 314-467 行)

#### 6.1 当前实现（错误的）

```python
def _process_and_split_data(self, df):
    """
    ❌ 问题：生成稀疏数据，每个样本只有一个节点有值
    """
    
    # 按节点对分组
    grouped = df.groupby(['nds_id', 'next_nds_id'])
    
    all_samples_X = []
    all_samples_Y = []
    
    # 🔴 问题开始：为每个节点单独生成样本
    for (nds_id, next_nds_id), group in grouped:
        node_idx = self.node_to_idx[(nds_id, next_nds_id)]
        
        # 按时间排序
        group = group.sort_values('passts_time')
        flow_series = group['flow_label'].values
        
        # 滑动窗口（只针对这一个节点）
        for i in range(len(flow_series) - input_len - output_len + 1):
            x = flow_series[i:i+input_len]  # (12,)
            y = flow_series[i+input_len:i+input_len+output_len]  # (12,)
            
            all_samples_X.append((node_idx, x))  # 🔴 只记录一个节点
            all_samples_Y.append((node_idx, y))
    
    # 构建数据数组
    X_data = np.zeros((num_samples, input_len, node_num, 1))
    Y_data = np.zeros((num_samples, output_len, node_num, 1))
    
    # 🔴 关键问题：每个样本只填充一个节点
    for i, ((node_idx, x), (_, y)) in enumerate(zip(all_samples_X, all_samples_Y)):
        X_data[i, :, node_idx, 0] = x  # 只有 node_idx 位置有值
        Y_data[i, :, node_idx, 0] = y  # 其余 12391 个位置都是 0
```

**问题分析**:
```
生成的数据格式:
X_data: (num_samples, 12, 12392, 1)

样本 0:
  时间步 0: [0, 0, 0, ..., 15, 0, 0, ...]  ← 只有节点 #3 有值
  时间步 1: [0, 0, 0, ..., 12, 0, 0, ...]
  ...
  
样本 1:
  时间步 0: [0, 8, 0, ..., 0, 0, 0, ...]   ← 只有节点 #1 有值
  时间步 1: [0, 10, 0, ..., 0, 0, 0, ...]
  ...

❌ 每个样本是稀疏的，只有 1/12392 的位置有值
```

#### 6.2 正确实现（应该的）

```python
def _process_and_split_data_correct(self, df):
    """
    ✅ 正确：按时间窗口组织，生成密集数据
    """
    
    # 步骤 1: 将数据 pivot 成时间序列格式
    # 从: (node, time, flow) 长格式
    # 到: (time, all_nodes) 宽格式
    
    # 按分钟对齐时间戳
    df['time_minute'] = df['passts_time'].dt.floor('1min')
    
    # Pivot: 行=时间，列=节点，值=流量
    time_series = df.pivot_table(
        index='time_minute',
        columns=['nds_id', 'next_nds_id'],
        values='flow_label',
        fill_value=0  # 缺失值填充为 0
    )
    
    # 现在 time_series: (10080 时间点, 12392 节点)
    
    # 步骤 2: 滑动窗口生成样本
    num_times = len(time_series)
    num_samples = num_times - input_len - output_len + 1
    
    X_data = np.zeros((num_samples, input_len, node_num, 1))
    Y_data = np.zeros((num_samples, output_len, node_num, 1))
    
    for i in range(num_samples):
        # 所有节点在连续时间步的数据
        X_data[i, :, :, 0] = time_series.iloc[i:i+input_len].values
        Y_data[i, :, :, 0] = time_series.iloc[i+input_len:i+input_len+output_len].values
    
    # ✅ 每个样本包含所有节点的数据（密集）
```

**正确的数据格式**:
```
X_data: (num_samples, 12, 12392, 1)

样本 0 (时间窗口: 08:00-08:11):
  时间步 0 (08:00): [15, 8, 0, 12, 5, ...]  ← 所有 12392 个节点都有值
  时间步 1 (08:01): [12, 10, 3, 15, 7, ...]
  时间步 2 (08:02): [10, 12, 5, 18, 8, ...]
  ...
  时间步 11 (08:11): [8, 15, 7, 20, 10, ...]

样本 1 (时间窗口: 08:01-08:12):
  时间步 0 (08:01): [12, 10, 3, 15, 7, ...]  ← 所有节点都有值
  时间步 1 (08:02): [10, 12, 5, 18, 8, ...]
  ...

✅ 每个样本是密集的，100% 的位置有值
```

#### 6.3 数据划分

```python
# 计算归一化参数（基于训练集）
num_train = int(num_samples * 0.7)
self.mean = np.mean(X_data[:num_train])
self.std = np.std(X_data[:num_train])

# 划分数据集
self.trainX = X_data[:num_train]
self.valX = X_data[num_train:num_train+num_val]
self.testX = X_data[num_train+num_val:]
```

---

### 阶段 7: 创建空间 Patch（必须有位置信息）

**文件**: `lib/odps_data_loader.py` → `_create_spatial_patches()` (第 469-544 行)

```python
def _create_spatial_patches(self, train_data):
    """
    使用 KD-tree 将节点划分为空间 patch
    
    ⚠️ 前提条件：必须已加载节点位置信息
    
    目的: 将 12392 个节点分组，每组约 spa_patchsize 个节点
    
    处理:
    1. 检查节点位置是否有效
    2. 使用节点位置 (lat, lng) 构建 KD-tree
    3. 递归分割（recur_times 次）
    4. 生成 patch 索引列表
    
    输出:
    - self.ori_parts_idx: [[0,1,2,3], [4,5,6,7], ...]  # 原始分组
    - self.reo_parts_idx: 重排后的分组（根据邻接矩阵）
    - self.reo_all_idx: 所有节点的重排索引
    """
```

**Patch 示例**:
```
假设 spa_patchsize = 4, recur_times = 2

步骤 1: 按纬度分割
  左半部分: 节点 0-6191
  右半部分: 节点 6192-12391

步骤 2: 每半部分再按经度分割
  Patch 1: 节点 [0, 15, 23, 45]      ← 空间上相近的节点
  Patch 2: 节点 [1, 18, 30, 52]
  ...
  Patch 3098: 节点 [12380, 12385, 12388, 12390]

总共: 12392 / 4 = 3098 个 patch
```

**为什么必须有位置信息？**
- KD-tree 分组基于节点的空间距离
- 将地理位置相近的节点分在同一个 patch
- 有助于模型学习空间相关性
- 如果没有位置信息，无法进行合理的空间分组

---

### 阶段 8: 训练模型

**文件**: `train_odps.py` → `ODPSSolver.train()` (第 120-173 行)

#### 8.1 构建模型

```python
def build_model(self):
    self.model = PatchSTG(
        output_len=12,           # 预测 12 个时间步
        tem_patchsize=4,         # 时间 patch 大小
        tem_patchnum=3,          # 时间 patch 数量 (12/4)
        node_num=12392,          # 节点数
        spa_patchsize=4,         # 空间 patch 大小
        spa_patchnum=3098,       # 空间 patch 数量 (12392/4)
        ...
    )
```

#### 8.2 训练循环

```python
def train(self):
    for epoch in range(1, max_epoch + 1):
        # 打乱训练数据
        self.data_loader.shuffle_train_data()
        
        # Mini-batch 训练
        for batch_idx in range(num_batch):
            # 获取 batch 数据
            X = self.trainX[start_idx:end_idx]  # (batch, 12, 12392, 1)
            Y = self.trainY[start_idx:end_idx]  # (batch, 12, 12392, 1)
            TE = self.trainXTE[start_idx:end_idx]  # (batch, 12, 2)
            
            # 归一化
            NormX = (X - self.mean) / self.std
            
            # 前向传播
            y_hat = self.model(NormX, TE)
            
            # 计算损失
            loss = _compute_loss(Y, y_hat * self.std + self.mean)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
        
        # 验证
        mae, rmse, mape = self.vali()
```

---

## 📈 数据形状变化追踪

```
ODPS 表记录:
  27,545,086 条 × (nds_id, next_nds_id, passts_time, flow_label)
  ↓

DataFrame:
  27,545,086 行 × 8 列
  ↓

节点列表:
  12,392 个唯一节点对
  ↓

❌ 当前实现（错误）:
  样本生成: 每个节点单独生成 → ~270万个样本
  X_data: (2,700,000, 12, 12392, 1)  ← 稀疏！
  ↓

✅ 正确实现（应该）:
  时间序列: (10080 时间点, 12392 节点)
  样本生成: 滑动窗口 → ~10,000个样本
  X_data: (10,057, 12, 12392, 1)  ← 密集！
  ↓

划分数据集:
  trainX: (7,040, 12, 12392, 1)  - 70%
  valX:   (1,006, 12, 12392, 1)  - 10%
  testX:  (2,011, 12, 12392, 1)  - 20%
  ↓

Mini-batch:
  batch_size = 64
  X: (64, 12, 12392, 1)
  Y: (64, 12, 12392, 1)
  TE: (64, 12, 2)
  ↓

模型输出:
  y_hat: (64, 12, 12392, 1)
```

---

## ⚠️ 核心问题总结

### 当前实现的问题

1. **数据组织错误**:
   - 按节点分组 → 每个样本只有一个节点
   - 应该按时间分组 → 每个样本包含所有节点

2. **稀疏性问题**:
   - 当前: 每个样本 1/12392 位置有值（0.008%）
   - 应该: 每个样本所有位置都有值（100%）

3. **样本数量错误**:
   - 当前: ~270万个样本（每个节点 ~220个样本）
   - 应该: ~1万个样本（时间窗口数量）

4. **内存浪费**:
   - 当前: 存储大量稀疏零值
   - 应该: 存储密集有效数据

### 修复方案

需要修改 `_process_and_split_data()` 方法：
- **从**: 按节点分组 → 节点级样本 → 稀疏矩阵
- **到**: 按时间分组 → 时间窗口样本 → 密集矩阵

---

## 🎯 下一步

**需要我现在修改 `lib/odps_data_loader.py` 中的 `_process_and_split_data()` 方法吗？**

修改内容:
1. 使用 `pivot_table` 将数据转换为时间序列格式
2. 在时间序列上使用滑动窗口生成样本
3. 确保每个样本包含所有节点的数据
4. 添加详细的断言和日志

修改后:
- ✅ 数据格式正确（密集矩阵）
- ✅ 样本数量合理（~1万个）
- ✅ 内存占用小（无稀疏零值）
- ✅ 可以正常训练
