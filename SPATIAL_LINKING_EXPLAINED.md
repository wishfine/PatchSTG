# 训练时节点经纬度与车流信息的关联机制

## 核心问题

**如何将地理位置信息（经纬度）与车流量数据关联起来进行空间建模？**

## 数据关联流程

### 1. 数据加载阶段

#### 步骤 1.1：加载主表车流数据
```python
# 从 ODPS 主表查询车流数据
SELECT 
    nds_id,           # 转向前的路段ID
    next_nds_id,      # 转向后的路段ID
    passts_time,      # 时间戳
    flow_label,       # 车流量值
    ...
FROM tb_inter_spatial_method_pretrain_data
WHERE adcode = '110000' AND ds >= '20250701' AND ds <= '20250731'
```

**结果示例**：
```
nds_id | next_nds_id | passts_time        | flow_label
-------|-------------|-------------------|------------
A001   | B002        | 2025-07-01 08:00 | 15
A001   | B002        | 2025-07-01 08:01 | 18
C003   | D004        | 2025-07-01 08:00 | 25
...
```

#### 步骤 1.2：构建节点列表
```python
# 提取所有唯一的转向流
node_list = df[['nds_id', 'next_nds_id']].drop_duplicates()

# 结果：[(A001, B002), (C003, D004), (E005, F006), ...]
# 每个元组代表一个转向流
```

**关键**：给每个转向流分配索引
```python
self.node_to_idx = {
    ('A001', 'B002'): 0,
    ('C003', 'D004'): 1,
    ('E005', 'F006'): 2,
    ...
}
```

#### 步骤 1.3：加载元数据表获取经纬度
```python
# 从元数据表查询位置信息
SELECT 
    nds_id,
    next_nds_id,
    inter_id,         # 路口ID
    latitude,         # 路口纬度
    longitude         # 路口经度
FROM tb_inter_spatial_node_location
WHERE adcode = '110000'
```

**结果示例**：
```
nds_id | next_nds_id | inter_id | latitude | longitude
-------|-------------|----------|----------|----------
A001   | B002        | I001     | 39.9042  | 116.4074  # 北京某路口
C003   | D004        | I002     | 39.9150  | 116.4050
E005   | F006        | I001     | 39.9042  | 116.4074  # 同一路口的另一个转向
...
```

### 2. 位置信息关联

#### 步骤 2.1：创建位置数组
```python
# 初始化位置数组：(2维, 转向流数量)
self.node_locations = np.zeros((2, self.node_num), dtype=np.float32)
#                               ↑            ↑
#                        [纬度, 经度]   转向流数量（如850个）
```

#### 步骤 2.2：填充每个转向流的位置
```python
# 遍历每个转向流
for idx, (nds_id, next_nds_id) in enumerate(self.node_list):
    # 查找该转向流对应的路口位置
    location = meta_df[
        (meta_df['nds_id'] == nds_id) & 
        (meta_df['next_nds_id'] == next_nds_id)
    ]
    
    if len(location) > 0:
        # 将路口的经纬度赋给这个转向流
        self.node_locations[0, idx] = location.iloc[0]['latitude']   # 纬度
        self.node_locations[1, idx] = location.iloc[0]['longitude']  # 经度
```

**结果**：位置数组已填充
```python
self.node_locations = [
    [39.9042, 39.9150, 39.9042, ...],  # 纬度数组
    [116.4074, 116.4050, 116.4074, ...]  # 经度数组
]
# 索引 0 → 转向流 (A001, B002) → 路口 I001 → (39.9042, 116.4074)
# 索引 1 → 转向流 (C003, D004) → 路口 I002 → (39.9150, 116.4050)
# 索引 2 → 转向流 (E005, F006) → 路口 I001 → (39.9042, 116.4074)
```

### 3. 车流数据处理

#### 步骤 3.1：构建车流量数组
```python
# 创建数据数组：(样本数, 时间步, 转向流数, 特征维度)
X_data = np.zeros((num_samples, input_len, node_num, 1))
#                  ↑           ↑          ↑         ↑
#                样本数      输入12步   850个转向流  1维特征(流量)

# 填充每个转向流的车流数据
for i, (node_idx, x) in enumerate(all_samples_X):
    X_data[i, :, node_idx, 0] = x  # 在对应索引位置填充流量值
```

**示例**：某一个样本的数据
```python
X_data[0] = [
    # 时间步0   时间步1   时间步2   ...
    [[15],      [18],     [20],     ...],  # 索引0：转向流(A001,B002)的流量
    [[25],      [22],     [28],     ...],  # 索引1：转向流(C003,D004)的流量
    [[10],      [12],     [11],     ...],  # 索引2：转向流(E005,F006)的流量
    ...
]
```

### 4. 空间Patching（关键关联）

#### 步骤 4.1：使用KD-tree进行空间聚类
```python
from sklearn.neighbors import KDTree

# 将位置数组转置：(转向流数, 2维坐标)
tree = KDTree(self.node_locations.T)
#              ↑
#      [[39.9042, 116.4074],   # 转向流0的位置
#       [39.9150, 116.4050],   # 转向流1的位置
#       [39.9042, 116.4074],   # 转向流2的位置
#       ...]
```

#### 步骤 4.2：递归划分形成Patch
```python
# 根据经纬度递归分割节点
def recursive_split(indices, depth=0):
    if depth >= recur_times or len(indices) <= spa_patchsize:
        return [indices]
    
    # 选择维度（偶数层用纬度，奇数层用经度）
    axis = depth % 2
    coords = self.node_locations[:, indices]
    
    # 按照选定维度排序并从中位数分割
    sorted_indices = indices[np.argsort(coords[axis, :])]
    median_idx = len(sorted_indices) // 2
    
    left = sorted_indices[:median_idx]
    right = sorted_indices[median_idx:]
    
    # 递归处理
    return recursive_split(left, depth+1) + recursive_split(right, depth+1)

# 执行划分
parts_idx = recursive_split(np.arange(node_num))
```

**划分结果示例**：
```python
parts_idx = [
    [0, 2, 5, 8],      # Patch 0: 这4个转向流地理位置相近（同一区域）
    [1, 3, 7, 9],      # Patch 1: 这4个转向流在另一个区域
    [4, 6, 10, 11],    # Patch 2: 第三个区域
    ...
]
```

**地理意义**：
- Patch 0 可能包含：某个路口的多个转向流 + 附近路口的转向流
- 地理上相近的转向流被分到同一个Patch中
- 模型可以学习局部空间依赖关系

### 5. 训练时的数据流

#### 输入到模型的数据结构
```python
# 训练数据
trainX: (batch_size, input_len, node_num, 1)
#       32个样本    12个时间步  850个节点  流量特征

# 节点位置（用于空间attention）
node_locations: (2, node_num)
#               ↑   ↑
#            [纬度  转向流数量
#             经度]

# Patch索引（用于分组处理）
reo_parts_idx: [[节点0, 节点2, 节点5, 节点8],      # Patch 0
                [节点1, 节点3, 节点7, 节点9],      # Patch 1
                ...]
```

#### 模型中的空间建模
```python
# 伪代码：模型处理流程
for patch_indices in reo_parts_idx:
    # 提取这个patch的节点数据和位置
    patch_data = trainX[:, :, patch_indices, :]
    patch_locations = node_locations[:, patch_indices]
    
    # 基于位置计算空间注意力权重
    spatial_attn = compute_spatial_attention(patch_locations)
    
    # 使用注意力权重聚合空间特征
    spatial_features = spatial_attn @ patch_data
    
    # ... 继续时间建模等
```

## 关联机制总结

### 关键映射关系

```
转向流 (nds_id, next_nds_id) 
    ↓ [node_to_idx 映射]
节点索引 idx
    ↓ [关联两个数组]
    ├─→ node_locations[0, idx] = 纬度  }  空间信息
    ├─→ node_locations[1, idx] = 经度  }
    └─→ X_data[:, :, idx, 0] = 流量序列   时序信息
```

### 具体示例

**转向流 (A001, B002)**:
- 索引: `idx = 0`
- 位置: `node_locations[:, 0] = [39.9042, 116.4074]`
- 流量: `X_data[sample_i, :, 0, 0] = [15, 18, 20, ...]`

**模型计算时**:
1. 找到该转向流所在的Patch（例如Patch 3）
2. 获取Patch 3内所有节点的位置：`node_locations[:, patch_3_indices]`
3. 获取Patch 3内所有节点的流量：`X_data[:, :, patch_3_indices, :]`
4. 基于位置计算空间依赖性，基于流量进行预测

## 为什么这样设计？

### 1. 保持索引一致性
- 车流数据和位置数据使用**相同的索引系统**
- 索引 `i` 永远对应同一个转向流
- 确保空间信息和时序信息不会错位

### 2. 支持稀疏数据
```python
X_data = np.zeros((samples, time, nodes, 1))
# 大部分位置是0（某个时刻某个转向流可能没有数据）
# 只填充有数据的转向流位置
```

### 3. 空间局部性
- KD-tree基于经纬度将地理相近的转向流分组
- 同一路口的不同转向流可能在同一个Patch
- 相邻路口的转向流也可能在同一个Patch
- 模型可以学习局部交通流动模式

### 4. 高效计算
- Patch化后，可以并行处理多个空间区域
- 每个Patch的节点数量受控（≈4个）
- 减少全局注意力计算的复杂度

## 数据流向图

```
ODPS主表                    元数据表
   ↓                          ↓
车流量记录                 路口位置信息
   ↓                          ↓
(nds_id, next_nds_id) ←──── inter_id + (lat, lng)
   ↓                          ↓
node_to_idx 映射        node_locations 数组
   ↓                          ↓
   └────── 统一的索引 ────────┘
              ↓
        X_data[samples, time, nodes, features]
              ↓
         KD-tree 空间划分
              ↓
     Patch分组 + 空间Attention
              ↓
          模型训练
```

## 验证关联是否正确

### 检查点1：索引映射
```python
# 选择一个转向流
test_node = ('A001', 'B002')
idx = self.node_to_idx[test_node]

print(f"转向流 {test_node}:")
print(f"  索引: {idx}")
print(f"  位置: 纬度={node_locations[0, idx]}, 经度={node_locations[1, idx]}")
print(f"  第一个样本的流量: {X_data[0, :, idx, 0]}")
```

### 检查点2：Patch地理连续性
```python
# 检查某个patch内的节点是否地理相近
patch_0_indices = reo_parts_idx[0]
patch_0_locations = node_locations[:, patch_0_indices]

print(f"Patch 0 的节点位置:")
for i, idx in enumerate(patch_0_indices):
    lat, lng = patch_0_locations[0, i], patch_0_locations[1, i]
    print(f"  节点{idx}: ({lat:.4f}, {lng:.4f})")

# 计算patch内节点之间的距离
from scipy.spatial.distance import pdist
distances = pdist(patch_0_locations.T)
print(f"Patch内最大距离: {distances.max():.4f} 度")
```

### 检查点3：数据完整性
```python
# 检查有多少转向流缺少位置信息
missing_locations = np.sum(
    (node_locations[0, :] == 0) & (node_locations[1, :] == 0)
)
coverage = (node_num - missing_locations) / node_num * 100
print(f"位置信息覆盖率: {coverage:.2f}%")
```

## 常见问题

### Q1: 多个转向流可以有相同的经纬度吗？
**A**: 可以！同一个路口有多个转向方向（左转、右转、直行等），它们共享路口的经纬度。

### Q2: 如果某个转向流缺少位置信息怎么办？
**A**: 
- 位置设为 (0, 0)
- 在KD-tree划分时会被分到一起
- 或者通过邻近插值填充

### Q3: 训练时会修改node_locations吗？
**A**: 不会！`node_locations`是固定的地理信息，不参与训练更新。

### Q4: 如果不提供元数据表会怎样？
**A**: 
- `use_spatial_patching = False`
- 所有节点放在一个大的Patch中
- 退化为简单的全局建模，失去空间局部性优势

## 总结

**核心思想**：通过统一的索引系统，将转向流的三类信息绑定在一起：
1. **身份信息**：(nds_id, next_nds_id)
2. **空间信息**：经纬度 → 用于KD-tree分组
3. **时序信息**：车流量时间序列 → 用于预测

这种设计保证了空间建模和时序建模的一致性，是PatchSTG模型能够同时利用时空信息的基础。
