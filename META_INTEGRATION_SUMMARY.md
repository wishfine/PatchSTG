# 元数据表集成总结

## 更新内容

为了支持 PatchSTG 的核心特性——**基于地理位置的空间 patching（KD-tree 划分）**，我已经更新了代码以支持从 ODPS 加载节点的地理位置信息。

## 关键改进

### 1. 支持元数据表加载

**文件**: `lib/odps_data_loader.py`

#### 新增配置参数
```python
self.odps_meta_table = config.get('odps_meta_table', None)  # 元数据表名
self.recur_times = config.get('recur_times', 1)  # KD-tree 递归次数
self.spa_patchsize = config.get('spa_patchsize', 4)  # 空间 patch 大小
self.node_locations = None  # 节点经纬度 (2, num_nodes): [lat, lng]
self.use_spatial_patching = self.odps_meta_table is not None
```

#### 新增方法

##### `_load_node_locations()`
- 从 ODPS 元数据表加载节点的经纬度信息
- 构建 `node_locations` 数组：`(2, num_nodes)` [lat, lng]
- 处理缺失数据的情况

```python
def _load_node_locations(self):
    """
    从 ODPS 元数据表加载节点的经纬度信息
    
    元数据表应包含:
    - nds_id: 起始节点 ID
    - next_nds_id: 终止节点 ID
    - latitude: 纬度
    - longitude: 经度
    """
```

##### `_create_spatial_patches(train_data)`
- 如果有位置信息，使用 KD-tree 进行空间划分
- 否则使用简单的顺序 patching

```python
def _create_spatial_patches(self, train_data):
    """
    创建空间 patch 索引
    
    如果有节点位置信息，使用 KD-tree 进行空间划分；
    否则使用简单的顺序划分
    """
```

##### `_create_simple_patches()`
- 创建简单的顺序 patch（备用方案）

### 2. KD-tree 空间划分

使用递归的 KD-tree 算法将节点按地理位置划分：

```python
def recursive_split(indices, depth=0):
    if depth >= self.recur_times or len(indices) <= self.spa_patchsize:
        return [indices]
    
    # 找到中位数并分割
    coords = self.node_locations[:, indices]
    axis = depth % 2  # 0 for lat, 1 for lng
    median_idx = len(indices) // 2
    sorted_indices = indices[np.argsort(coords[axis, :])]
    
    left = sorted_indices[:median_idx]
    right = sorted_indices[median_idx:]
    
    return recursive_split(left, depth+1) + recursive_split(right, depth+1)
```

### 3. 配置文件更新

**文件**: `config/ODPS.conf`

添加元数据表配置：

```ini
[data]
# ...

# ODPS 元数据表（可选，包含节点经纬度信息）
# 如果提供此表，将使用 KD-tree 进行真实的空间 patching
# 表结构应包含: nds_id, next_nds_id, latitude, longitude
# 留空则使用简单的顺序 patching
odps_meta_table = tb_inter_spatial_node_location

# ...
```

### 4. 文档

#### 新增文档

**`META_TABLE_GUIDE.md`** - 元数据表完整指南
- 为什么需要元数据表
- 表结构定义
- 如何计算路段的代表点位置
- 创建元数据表的 SQL 示例
- 数据质量检查
- 配置使用说明
- 常见问题解答

#### 更新文档

**`ODPS_TRAINING_GUIDE.md`**
- 添加元数据表的说明
- 更新字段说明章节

**`README.md`**
- 添加元数据表的使用说明
- 添加文档链接

## 使用方式

### 方式 1: 使用元数据表（推荐）

1. **创建元数据表**

参考 `META_TABLE_GUIDE.md` 创建包含节点经纬度的表：

```sql
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_node_location AS
SELECT 
    nds_id,
    next_nds_id,
    adcode,
    (start_lat + end_lat) / 2.0 AS latitude,
    (start_lng + end_lng) / 2.0 AS longitude
FROM your_road_network_table;
```

2. **配置元数据表**

编辑 `config/ODPS.conf`:
```ini
odps_meta_table = tb_inter_spatial_node_location
```

3. **训练**

```bash
python train_odps.py --config config/ODPS.conf --mode train
```

**预期输出**:
```
✓ Loading node locations from table: tb_inter_spatial_node_location
✓ Loaded locations for 850 nodes
✓ Creating spatial patches using KD-tree...
✓ Created 16 spatial patches
✓ Patch sizes: [52, 54, 51, 53, ...]
```

### 方式 2: 不使用元数据表（简化版）

1. **配置留空**

编辑 `config/ODPS.conf`:
```ini
odps_meta_table = 
```

2. **训练**

```bash
python train_odps.py --config config/ODPS.conf --mode train
```

**预期输出**:
```
ℹ No meta table specified, skipping location loading
ℹ Using simple sequential patching (no location data)
```

## 元数据表结构

### 必需字段

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| nds_id | STRING | 起始节点 ID | '123456' |
| next_nds_id | STRING | 终止节点 ID | '789012' |
| latitude | DOUBLE | 纬度 | 39.9042 |
| longitude | DOUBLE | 经度 | 116.4074 |

### 可选字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| adcode | STRING | 行政区划代码（用于过滤） |
| road_name | STRING | 道路名称 |

## 数据流程

### 有元数据表的流程

```
1. 加载流量数据
   ↓
2. 构建节点列表 (nds_id, next_nds_id)
   ↓
3. 从元数据表加载经纬度
   ↓
4. 使用 KD-tree 进行空间划分
   ↓
5. 生成训练样本
   ↓
6. 训练模型（使用空间 patching）
```

### 无元数据表的流程

```
1. 加载流量数据
   ↓
2. 构建节点列表 (nds_id, next_nds_id)
   ↓
3. 使用简单顺序 patching
   ↓
4. 生成训练样本
   ↓
5. 训练模型（所有节点在一个 patch）
```

## 性能对比

| 特性 | 有元数据表 | 无元数据表 |
|------|-----------|-----------|
| 空间划分 | KD-tree | 顺序 |
| Patch 数量 | 多个（2^recur） | 1 个 |
| 空间局部性 | ✅ 保留 | ❌ 丢失 |
| 模型性能 | ✅ 最优 | ⚠️ 降低 |
| 计算复杂度 | ✅ 降低 | ⚠️ 较高 |
| 实现难度 | 需要额外表 | 简单 |

## 与原 NPZ 数据的对比

| 特性 | NPZ 数据 | ODPS (有元数据) | ODPS (无元数据) |
|------|----------|----------------|----------------|
| 节点定义 | 单个交叉口 | 有向路段 | 有向路段 |
| 位置信息 | meta.csv 文件 | ODPS 元数据表 | 无 |
| 空间划分 | KD-tree | KD-tree | 简单 |
| 邻接关系 | 预计算文件 | 动态构建 | 相似度矩阵 |

## 验证方法

### 1. 检查日志输出

**有元数据表**:
```
✓ Loading node locations from table: tb_inter_spatial_node_location
✓ Loaded locations for 850 nodes
✓ Creating spatial patches using KD-tree...
✓ Created 16 spatial patches
```

**无元数据表**:
```
ℹ No meta table specified, skipping location loading
ℹ Using simple sequential patching (no location data)
```

### 2. 运行数据检查

```bash
python check_odps_data.py --config config/ODPS.conf
```

### 3. 查看数据信息

```python
from lib.odps_data_loader import ODPSDataLoader

data_loader = ODPSDataLoader(config, log)
data_loader.load_data()

info = data_loader.get_data_info()
print(f"Spatial patching: {data_loader.use_spatial_patching}")
print(f"Patches: {len(data_loader.ori_parts_idx)}")
```

## 注意事项

### 1. 数据一致性

- 确保元数据表覆盖所有流量表中的路段
- 检查 nds_id 和 next_nds_id 的数据类型一致性
- 验证经纬度的合理性

### 2. 性能权衡

- **建议**: 如果有条件，强烈建议创建元数据表
- **临时**: 如果暂时无法创建，可以先用简单模式
- **后续**: 可以随时添加元数据表并重新训练

### 3. 坐标系统

- 推荐使用 WGS84（GPS 坐标系）
- 所有节点必须使用相同的坐标系统
- 纬度范围: -90 ~ 90
- 经度范围: -180 ~ 180

### 4. 错误处理

代码包含完整的错误处理：
- 元数据表不存在 → 自动降级到简单 patching
- 部分节点位置缺失 → 使用默认值 (0, 0)
- KD-tree 划分失败 → 降级到简单 patching

## 下一步

### 如果你有节点位置信息

1. 参考 `META_TABLE_GUIDE.md` 创建元数据表
2. 更新配置文件指定表名
3. 运行 `check_odps_data.py` 验证
4. 开始训练

### 如果暂时没有位置信息

1. 配置文件中将 `odps_meta_table` 留空
2. 直接开始训练（使用简单 patching）
3. 后续有条件时再补充元数据表
4. 重新训练以获得更好性能

## 总结

✅ **已实现**:
- 从 ODPS 元数据表加载节点位置
- 使用 KD-tree 进行空间划分
- 自动降级机制（无元数据表时）
- 完整的错误处理
- 详细的文档和示例

✅ **向后兼容**:
- 不提供元数据表时仍可正常运行
- 与原有代码完全兼容

✅ **性能优化**:
- 使用地理位置进行智能划分
- 保留空间局部性
- 降低计算复杂度

🎯 **推荐做法**:
- 优先创建元数据表以获得最佳性能
- 参考 `META_TABLE_GUIDE.md` 了解详细步骤
- 定期维护元数据表以保持数据质量
