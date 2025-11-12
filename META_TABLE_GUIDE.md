# ODPS 元数据表使用指南

## 概述

为了实现 PatchSTG 的核心特性——**基于地理位置的空间 patching（KD-tree 划分）**，你需要提供一个包含路段地理位置信息的元数据表。

## 为什么需要元数据表？

PatchSTG 模型的一个关键创新是使用 **KD-tree 对节点进行空间划分**，将地理上相近的路段划分到同一个 patch 中。这样可以：

1. ✅ **提高效率**: 减少 Attention 机制的计算复杂度
2. ✅ **保留空间关联**: 地理相近的路段通常有相似的交通模式
3. ✅ **平衡 patch 大小**: 确保每个 patch 的节点数相对均衡

**如果没有元数据表**，系统会退化到简单的顺序 patching（将所有节点放在一个 patch 中），这会降低模型性能。

## 元数据表结构

### 概念说明

**关键理解**：
- `nds_id` = 转向前的路段 ID（进入该路口的路段）
- `next_nds_id` = 转向后的路段 ID（离开该路口的路段）
- `(nds_id, next_nds_id)` = 一个转向流，对应一个路口 `inter_id`
- `inter_id` = 路口 ID，这个路口有经纬度坐标

**示例**：
```
路段 A (nds_id=123) → 路口 X (inter_id=999, lat=39.90, lng=116.40) → 路段 B (next_nds_id=456)
```

这个转向流 `(123, 456)` 对应的路口是 `inter_id=999`，位置在 `(39.90, 116.40)`。

### 表名建议
```
autonavi_traffic_report.tb_inter_spatial_node_location
```

### 必需字段

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `nds_id` | STRING | 转向前的路段 ID | '123456' |
| `next_nds_id` | STRING | 转向后的路段 ID | '789012' |
| `inter_id` | STRING | 路口 ID | '999888' |
| `latitude` | DOUBLE | 路口纬度 | 39.9042 |
| `longitude` | DOUBLE | 路口经度 | 116.4074 |

### 可选字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `adcode` | STRING | 行政区划代码（用于过滤） |
| `inter_name` | STRING | 路口名称 |
| `district` | STRING | 所属区域 |

## 如何获取路口位置信息？

由于每个转向流 `(nds_id, next_nds_id)` 对应一个路口 `inter_id`，你需要从路网数据中获取这个映射关系。

### 前提条件

你需要有以下数据源之一：

1. **路网拓扑表**：包含路段之间的连接关系和路口信息
2. **路口表**：包含所有路口的 ID 和经纬度
3. **转向关系表**：记录了 `(nds_id, next_nds_id)` → `inter_id` 的映射

### 典型的数据关系

### 典型的数据关系

```
路段表 (road_segment)               路口表 (intersection)
┌─────────┬──────────┐            ┌──────────┬──────┬──────┐
│ nds_id  │ end_node │            │ inter_id │ lat  │ lng  │
├─────────┼──────────┤            ├──────────┼──────┼──────┤
│ 123456  │ node_A   │            │ 999888   │ 39.9 │116.4 │
│ 789012  │ node_B   │            │ 777666   │ 40.0 │116.5 │
└─────────┴──────────┘            └──────────┴──────┴──────┘
         │                                │
         └────────┬───────────────────────┘
                  │
    转向关系表 (turn_relation)
    ┌─────────┬──────────────┬──────────┐
    │ nds_id  │ next_nds_id  │ inter_id │
    ├─────────┼──────────────┼──────────┤
    │ 123456  │ 789012       │ 999888   │  ← 从路段123456经过路口999888转向到路段789012
    └─────────┴──────────────┴──────────┘
```

## 创建元数据表的 SQL 示例

### 方法 1: 从转向关系表创建（推荐）

如果你有完整的转向关系表和路口表：

```sql
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_node_location AS
SELECT 
    t.nds_id,
    t.next_nds_id,
    t.inter_id,
    i.latitude,
    i.longitude,
    t.adcode
FROM turn_relation_table t
INNER JOIN intersection_table i
    ON t.inter_id = i.inter_id
WHERE t.adcode IN ('110000', '310000', '440100', '440300')  -- 可选：过滤特定城市
;
```

### 方法 2: 从路网拓扑推导

如果路段表中包含起止节点信息：

```sql
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_node_location AS
SELECT 
    r1.nds_id,
    r2.nds_id AS next_nds_id,
    r1.end_node AS inter_id,  -- 第一条路的终点 = 第二条路的起点 = 路口
    n.latitude,
    n.longitude,
    r1.adcode
FROM road_segment_table r1
INNER JOIN road_segment_table r2
    ON r1.end_node = r2.start_node  -- 两条路在同一个节点相连
    AND r1.adcode = r2.adcode
INNER JOIN node_table n
    ON r1.end_node = n.node_id
WHERE r1.adcode IN ('110000', '310000')
;
```

### 方法 3: 从流量表反推（如果没有专门的路网表）

```sql
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_node_location AS
SELECT DISTINCT
    f.nds_id,
    f.next_nds_id,
    -- 假设你有一个函数或子查询可以找到对应的路口ID
    get_intersection_id(f.nds_id, f.next_nds_id) AS inter_id,
    i.latitude,
    i.longitude,
    f.adcode
FROM autonavi_traffic_report.tb_inter_spatial_method_pretrain_data f
INNER JOIN intersection_table i
    ON get_intersection_id(f.nds_id, f.next_nds_id) = i.inter_id
WHERE f.ds >= '20250101'
;
```

### 方法 4: 如果只有路段端点坐标

如果路段表中有端点坐标，但没有明确的 inter_id：

```sql
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_node_location AS
SELECT 
    r1.nds_id,
    r2.nds_id AS next_nds_id,
    -- 使用路段端点的坐标作为路口位置
    -- 第一条路的终点 = 第二条路的起点 = 路口位置
    CONCAT(r1.nds_id, '_', r2.nds_id) AS inter_id,  -- 构造虚拟的路口ID
    r1.end_lat AS latitude,
    r1.end_lng AS longitude,
    r1.adcode
FROM road_segment_table r1
INNER JOIN road_segment_table r2
    ON r1.end_lat = r2.start_lat 
    AND r1.end_lng = r2.start_lng
    AND r1.adcode = r2.adcode
WHERE r1.adcode IN ('110000')
;
```

## 数据质量检查

创建表后，运行以下查询检查数据质量：

### 1. 检查总记录数
```sql
SELECT COUNT(*) AS total_turns
FROM autonavi_traffic_report.tb_inter_spatial_node_location;
```

### 2. 检查唯一的路口数
```sql
SELECT 
    COUNT(DISTINCT inter_id) AS unique_intersections,
    COUNT(DISTINCT nds_id, next_nds_id) AS unique_turns,
    COUNT(*) AS total_records
FROM autonavi_traffic_report.tb_inter_spatial_node_location;
```

**说明**：
- `unique_intersections`: 不同的路口数量
- `unique_turns`: 不同的转向流数量
- 一个路口可能对应多个转向流（例如：可以左转、直行、右转）

### 2. 检查唯一的路口数
```sql
SELECT 
    COUNT(DISTINCT inter_id) AS unique_intersections,
    COUNT(DISTINCT nds_id, next_nds_id) AS unique_turns,
    COUNT(*) AS total_records
FROM autonavi_traffic_report.tb_inter_spatial_node_location;
```

**说明**：
- `unique_intersections`: 不同的路口数量
- `unique_turns`: 不同的转向流数量
- 一个路口可能对应多个转向流（例如：可以左转、直行、右转）

### 3. 检查缺失值
```sql
SELECT 
    COUNT(*) AS total,
    SUM(CASE WHEN inter_id IS NULL THEN 1 ELSE 0 END) AS missing_inter,
    SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) AS missing_lat,
    SUM(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) AS missing_lng
FROM autonavi_traffic_report.tb_inter_spatial_node_location;
```

### 4. 检查坐标范围（以北京为例）
```sql
SELECT 
    COUNT(*) AS total,
    SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) AS missing_lat,
    SUM(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) AS missing_lng
FROM autonavi_traffic_report.tb_inter_spatial_node_location;
```

### 4. 检查坐标范围（以北京为例）
```sql
SELECT 
    adcode,
    MIN(latitude) AS min_lat,
    MAX(latitude) AS max_lat,
    MIN(longitude) AS min_lng,
    MAX(longitude) AS max_lng,
    COUNT(DISTINCT inter_id) AS intersection_count,
    COUNT(*) AS turn_count
FROM autonavi_traffic_report.tb_inter_spatial_node_location
WHERE adcode = '110000'  -- 北京
GROUP BY adcode;
```
```sql
SELECT 
    adcode,
    MIN(latitude) AS min_lat,
    MAX(latitude) AS max_lat,
    MIN(longitude) AS min_lng,
    MAX(longitude) AS max_lng,
    COUNT(*) AS segment_count
FROM autonavi_traffic_report.tb_inter_spatial_node_location
WHERE adcode = '110000'  -- 北京
GROUP BY adcode;
```

**北京正常范围参考**:
- 纬度: 39.4° ~ 41.1°
- 经度: 115.4° ~ 117.5°

### 5. 检查与流量表的覆盖率
```sql
SELECT 
    COUNT(DISTINCT f.nds_id, f.next_nds_id) AS traffic_turns,
    COUNT(DISTINCT m.nds_id, m.next_nds_id) AS meta_turns,
    COUNT(DISTINCT CASE 
        WHEN m.nds_id IS NOT NULL THEN f.nds_id, f.next_nds_id 
    END) AS matched_turns,
    COUNT(DISTINCT CASE 
        WHEN m.nds_id IS NOT NULL THEN f.nds_id, f.next_nds_id 
    END) * 100.0 / COUNT(DISTINCT f.nds_id, f.next_nds_id) AS coverage_pct
FROM autonavi_traffic_report.tb_inter_spatial_method_pretrain_data f
LEFT JOIN autonavi_traffic_report.tb_inter_spatial_node_location m
    ON f.nds_id = m.nds_id 
    AND f.next_nds_id = m.next_nds_id
WHERE f.adcode = '110000'
  AND f.ds BETWEEN '20250701' AND '20250731';
```

**理想覆盖率**: >= 95%

### 6. 查看每个路口的转向数量分布
```sql
SELECT 
    turn_count,
    COUNT(*) AS intersection_count
FROM (
    SELECT 
        inter_id,
        COUNT(*) AS turn_count
    FROM autonavi_traffic_report.tb_inter_spatial_node_location
    WHERE adcode = '110000'
    GROUP BY inter_id
) t
GROUP BY turn_count
ORDER BY turn_count;
```

**典型分布**：
- 2个转向：简单路口（如 T 字路口）
- 3-4个转向：十字路口
- 5个以上：复杂路口
```sql
SELECT 
    COUNT(DISTINCT t1.nds_id, t1.next_nds_id) AS traffic_segments,
    COUNT(DISTINCT t2.nds_id, t2.next_nds_id) AS meta_segments,
    COUNT(DISTINCT t1.nds_id, t1.next_nds_id) * 1.0 / 
        COUNT(DISTINCT t2.nds_id, t2.next_nds_id) AS coverage_ratio
FROM autonavi_traffic_report.tb_inter_spatial_method_pretrain_data t1
LEFT JOIN autonavi_traffic_report.tb_inter_spatial_node_location t2
    ON t1.nds_id = t2.nds_id 
    AND t1.next_nds_id = t2.next_nds_id
WHERE t1.adcode = '110000'
  AND t1.ds BETWEEN '20250701' AND '20250731';
```

**理想覆盖率**: >= 95%

## 配置使用

### 1. 更新配置文件

编辑 `config/ODPS.conf`:

```ini
[data]
# ... 其他配置 ...

# 指定元数据表
odps_meta_table = tb_inter_spatial_node_location

# ... 其他配置 ...
```

### 2. 运行数据检查

```bash
python check_odps_data.py --config config/ODPS.conf
```

检查输出中应该显示：
```
✓ Loading node locations from table: tb_inter_spatial_node_location
✓ Loaded locations for XXX nodes
✓ Creating spatial patches using KD-tree...
✓ Created XX spatial patches
```

### 3. 开始训练

```bash
python train_odps.py --config config/ODPS.conf --mode train
```

## 没有元数据表怎么办？

如果暂时无法提供元数据表，系统会自动退化到**简单 patching 模式**：

```ini
[data]
# 留空或注释掉此行
# odps_meta_table = 
odps_meta_table = 
```

**简单 patching 的影响**:
- ⚠️ 所有节点放在一个 patch 中
- ⚠️ 失去空间局部性优势
- ⚠️ 可能降低模型性能
- ✅ 但模型仍然可以训练和运行

## 元数据表维护建议

### 1. 定期更新
```sql
-- 增量更新：添加新出现的路段
INSERT OVERWRITE TABLE tb_inter_spatial_node_location
SELECT * FROM (
    SELECT * FROM tb_inter_spatial_node_location_old
    UNION ALL
    SELECT DISTINCT
        nds_id, next_nds_id, adcode, 
        latitude, longitude
    FROM new_segments_table
) t;
```

### 2. 数据验证
- 定期检查坐标范围是否合理
- 验证与流量表的覆盖率
- 清理无效或异常的坐标

### 3. 性能优化
```sql
-- 添加分区（如果数据量大）
ALTER TABLE tb_inter_spatial_node_location 
ADD PARTITION (adcode='110000');

-- 创建索引（根据 MaxCompute 版本）
-- 或使用 clustered by 提高查询效率
```

## 示例：完整的建表流程

```sql
-- 步骤 1: 创建临时表，提取唯一路段
CREATE TABLE temp_unique_segments AS
SELECT DISTINCT
    nds_id,
    next_nds_id,
    adcode
FROM autonavi_traffic_report.tb_inter_spatial_method_pretrain_data
WHERE ds >= '20250101';

-- 步骤 2: 关联节点位置信息
CREATE TABLE temp_segments_with_location AS
SELECT 
    t.nds_id,
    t.next_nds_id,
    t.adcode,
    n1.latitude AS start_lat,
    n1.longitude AS start_lng,
    n2.latitude AS end_lat,
    n2.longitude AS end_lng
FROM temp_unique_segments t
LEFT JOIN your_node_location_source n1
    ON t.nds_id = CAST(n1.node_id AS STRING)
LEFT JOIN your_node_location_source n2
    ON t.next_nds_id = CAST(n2.node_id AS STRING);

-- 步骤 3: 计算路段中点并创建最终表
CREATE TABLE autonavi_traffic_report.tb_inter_spatial_node_location AS
SELECT 
    nds_id,
    next_nds_id,
    adcode,
    (start_lat + end_lat) / 2.0 AS latitude,
    (start_lng + end_lng) / 2.0 AS longitude,
    start_lat,
    start_lng,
    end_lat,
    end_lng
FROM temp_segments_with_location
WHERE start_lat IS NOT NULL 
  AND end_lat IS NOT NULL;

-- 步骤 4: 清理临时表
DROP TABLE temp_unique_segments;
DROP TABLE temp_segments_with_location;
```

## 常见问题

### Q1: 元数据表和流量表的节点数不一致怎么办？
**A**: 正常情况下，元数据表应该包含所有流量表中出现的路段。如果有缺失：
1. 检查 join 条件是否正确（数据类型、空格等）
2. 检查流量表的时间范围
3. 考虑使用历史数据补全缺失的路段

### Q2: 经纬度数据不准确怎么办？
**A**: 
1. 从高德或百度地图 API 获取准确坐标
2. 使用道路中心线的几何中心
3. 至少保证相对位置关系正确（KD-tree 主要利用相对位置）

### Q3: 可以使用其他坐标系统吗？
**A**: 可以，只要保证：
1. 所有节点使用相同的坐标系统
2. 坐标系统支持欧几里得距离计算
3. 推荐使用 WGS84（GPS 坐标系）

### Q4: 元数据表需要多久更新一次？
**A**: 
- **新增路段**: 当路网有新增路段时更新
- **定期维护**: 建议每季度检查一次数据质量
- **训练前**: 确保元数据表覆盖训练数据的时间范围

## 总结

元数据表是实现 PatchSTG 空间 patching 的关键：

✅ **推荐做法**: 
1. 创建包含所有路段经纬度的元数据表
2. 定期维护，确保数据质量
3. 使用 KD-tree 进行真实的空间划分

⚠️ **临时方案**: 
- 如果暂时无法提供，可以留空 `odps_meta_table`
- 系统会使用简单 patching，但性能会受影响

📖 **参考原论文**: PatchSTG 论文中详细描述了空间 patching 的重要性和实现细节
