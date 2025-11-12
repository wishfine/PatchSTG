-- ==========================================
-- ODPS 数据预处理脚本
-- 目标：将长格式数据转换为 PatchSTG 所需的时间序列格式
-- ==========================================

-- 说明：
-- 原始表: tb_inter_spatial_method_pretrain_data
--   每条记录 = 一个转向流在一个时刻的数据
--   字段: nds_id, next_nds_id, passts_time, flow_label, time_feat, dym_feat_feat
--
-- 目标表: tb_inter_traffic_timeseries
--   每条记录 = 一个时间窗口的所有转向流数据
--   字段: time_window, adcode, flow_matrix, time_features, node_list

-- ==========================================
-- 步骤 1: 创建时间窗口标签
-- ==========================================
-- 对于每个 passts_time，创建一个时间窗口标识符
-- 例如: 2025-09-19 08:15:30 -> window='2025-09-19 08:15'

DROP TABLE IF EXISTS tmp_time_windows;

CREATE TABLE tmp_time_windows AS
SELECT 
    -- 时间窗口（精确到分钟）
    CONCAT(
        ds, ' ',
        LPAD(CAST(hour AS STRING), 2, '0'), ':',
        LPAD(CAST(minute AS STRING), 2, '0')
    ) as time_window,
    
    -- 基本信息
    nds_id,
    next_nds_id,
    adcode,
    ds,
    
    -- 当前时刻的流量
    flow_label,
    
    -- 时间特征（取最后一段，即当前时刻）
    SPLIT(time_feat, ';')[SIZE(SPLIT(time_feat, ';')) - 1] as current_time_feat,
    
    -- 历史流量序列（取最后12个值作为输入序列）
    dym_feat_feat
    
FROM tb_inter_spatial_method_pretrain_data
WHERE adcode = '650100'  -- 乌鲁木齐
  AND ds >= '20250919'
  AND ds <= '20250925';


-- ==========================================
-- 步骤 2: 为每个时间窗口构建节点列表和索引
-- ==========================================
-- 创建全局节点列表（所有唯一的转向流）

DROP TABLE IF EXISTS tmp_node_list;

CREATE TABLE tmp_node_list AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY nds_id, next_nds_id) - 1 as node_idx,
    nds_id,
    next_nds_id
FROM (
    SELECT DISTINCT 
        nds_id,
        next_nds_id
    FROM tmp_time_windows
) t;


-- ==========================================
-- 步骤 3: 为每个时间窗口聚合所有节点的流量
-- ==========================================
-- 关键：使用 WM_CONCAT 将所有节点的流量拼接成字符串

DROP TABLE IF EXISTS tmp_window_aggregated;

CREATE TABLE tmp_window_aggregated AS
SELECT 
    w.time_window,
    w.adcode,
    w.ds,
    
    -- 时间特征（取任意一条记录，因为同一时间窗口的时间特征相同）
    MAX(w.current_time_feat) as time_features,
    
    -- 节点数量
    COUNT(DISTINCT CONCAT(CAST(w.nds_id AS STRING), '_', CAST(w.next_nds_id AS STRING))) as node_count,
    
    -- 流量矩阵：所有节点的流量按 node_idx 排序后拼接
    -- 格式: "idx1:flow1;idx2:flow2;idx3:flow3;..."
    WM_CONCAT(';', 
        CONCAT(
            CAST(n.node_idx AS STRING), 
            ':', 
            CAST(w.flow_label AS STRING)
        )
    ) as flow_matrix_sparse,
    
    -- 历史流量矩阵：所有节点的历史流量序列
    -- 格式: "idx1:val1,val2,...,val12;idx2:val1,val2,...,val12;..."
    WM_CONCAT(';',
        CONCAT(
            CAST(n.node_idx AS STRING),
            ':',
            w.dym_feat_feat
        )
    ) as history_matrix_sparse

FROM tmp_time_windows w
INNER JOIN tmp_node_list n
    ON w.nds_id = n.nds_id 
    AND w.next_nds_id = n.next_nds_id
GROUP BY 
    w.time_window,
    w.adcode,
    w.ds;


-- ==========================================
-- 步骤 4: 创建最终的预处理表
-- ==========================================

DROP TABLE IF EXISTS tb_inter_traffic_timeseries;

CREATE TABLE tb_inter_traffic_timeseries (
    time_window STRING COMMENT '时间窗口，格式: YYYYMMDD HH:MM',
    adcode STRING COMMENT '城市代码',
    time_features STRING COMMENT '时间特征，格式: week hour minute is_weekend day_of_week month',
    node_count BIGINT COMMENT '该时间窗口的节点数量',
    flow_matrix_sparse STRING COMMENT '流量矩阵（稀疏格式），格式: idx1:flow1;idx2:flow2;...',
    history_matrix_sparse STRING COMMENT '历史流量矩阵（稀疏格式），格式: idx1:h1,h2,...,h12;idx2:h1,h2,...,h12;...'
)
PARTITIONED BY (ds STRING)
COMMENT 'PatchSTG 训练数据 - 时间序列格式'
LIFECYCLE 365;

-- 插入数据
INSERT OVERWRITE TABLE tb_inter_traffic_timeseries PARTITION (ds)
SELECT 
    time_window,
    adcode,
    time_features,
    node_count,
    flow_matrix_sparse,
    history_matrix_sparse,
    ds
FROM tmp_window_aggregated;


-- ==========================================
-- 步骤 5: 创建节点元数据表（包含位置信息）
-- ==========================================

DROP TABLE IF EXISTS tb_inter_node_metadata;

CREATE TABLE tb_inter_node_metadata (
    node_idx BIGINT COMMENT '节点索引（从0开始）',
    nds_id BIGINT COMMENT 'NDS起点ID',
    next_nds_id BIGINT COMMENT 'NDS终点ID',
    inter_id STRING COMMENT '路口ID',
    lat DOUBLE COMMENT '纬度',
    lng DOUBLE COMMENT '经度',
    adcode STRING COMMENT '城市代码'
)
COMMENT 'PatchSTG 节点元数据'
LIFECYCLE 365;

-- 插入数据（关联位置信息）
INSERT OVERWRITE TABLE tb_inter_node_metadata
SELECT 
    n.node_idx,
    n.nds_id,
    n.next_nds_id,
    m.inter_id,
    m.lat,
    m.lng,
    m.adcode
FROM tmp_node_list n
LEFT JOIN intersection_meta_1 m
    ON n.nds_id = m.nds_id
    AND n.next_nds_id = m.next_nds_id;


-- ==========================================
-- 步骤 6: 数据质量检查
-- ==========================================

-- 检查1: 统计每个时间窗口的节点数
SELECT 
    '时间窗口节点数统计' as check_name,
    MIN(node_count) as min_nodes,
    MAX(node_count) as max_nodes,
    AVG(node_count) as avg_nodes,
    COUNT(*) as window_count
FROM tb_inter_traffic_timeseries
WHERE ds >= '20250919' AND ds <= '20250925';

-- 检查2: 统计总节点数
SELECT 
    '节点元数据统计' as check_name,
    COUNT(*) as total_nodes,
    SUM(CASE WHEN lat IS NOT NULL AND lng IS NOT NULL THEN 1 ELSE 0 END) as nodes_with_location,
    SUM(CASE WHEN lat IS NOT NULL AND lng IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as location_coverage_pct
FROM tb_inter_node_metadata;

-- 检查3: 查看样本数据
SELECT 
    time_window,
    node_count,
    SUBSTR(flow_matrix_sparse, 1, 100) as flow_sample,
    time_features
FROM tb_inter_traffic_timeseries
WHERE ds = '20250919'
LIMIT 5;


-- ==========================================
-- 清理临时表
-- ==========================================

DROP TABLE IF EXISTS tmp_time_windows;
DROP TABLE IF EXISTS tmp_node_list;
DROP TABLE IF EXISTS tmp_window_aggregated;


-- ==========================================
-- 使用说明
-- ==========================================

/*
预处理后的数据格式：

1. tb_inter_traffic_timeseries（时间序列数据）
   - time_window: '20250919 08:15'
   - adcode: '650100'
   - time_features: '5 8 15 0 4 9' (week, hour, minute, is_weekend, day_of_week, month)
   - node_count: 12392
   - flow_matrix_sparse: '0:15;1:8;2:0;3:12;...'  (node_idx:flow_value)
   - history_matrix_sparse: '0:5;3;2;1;0;0;8;15;12;10;8;6;1:...'

2. tb_inter_node_metadata（节点元数据）
   - node_idx: 0
   - nds_id: 123456
   - next_nds_id: 789012
   - inter_id: 'INT_001'
   - lat: 43.825
   - lng: 87.616
   - adcode: '650100'

在 Python 中使用：

from odps import ODPS

# 1. 读取节点元数据（只需读取一次）
query_meta = "SELECT * FROM tb_inter_node_metadata ORDER BY node_idx"
with odps.execute_sql(query_meta).open_reader() as reader:
    node_list = list(reader)  # [(node_idx, nds_id, next_nds_id, lat, lng), ...]

# 2. 读取时间序列数据
query_data = """
SELECT * FROM tb_inter_traffic_timeseries
WHERE ds >= '20250919' AND ds <= '20250925'
ORDER BY time_window
"""

with odps.execute_sql(query_data).open_reader() as reader:
    for record in reader:
        time_window = record['time_window']
        node_count = record['node_count']
        
        # 解析稀疏流量矩阵
        flow_dict = {}
        for item in record['flow_matrix_sparse'].split(';'):
            idx, flow = item.split(':')
            flow_dict[int(idx)] = float(flow)
        
        # 构建密集矩阵 (N, 1)
        N = len(node_list)
        flow_matrix = np.zeros((N, 1))
        for idx, flow in flow_dict.items():
            flow_matrix[idx, 0] = flow
        
        # 同样处理历史流量...
        
        # 现在 flow_matrix 是完整的 (N, 1) 格式！
*/
