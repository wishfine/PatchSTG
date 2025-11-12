"""
ODPS 数据加载器
从 MaxCompute 表加载交通流量数据进行训练
表名: autonavi_traffic_report.tb_inter_spatial_method_pretrain_data
"""
import os
import numpy as np
import pandas as pd
from odps import ODPS
from datetime import datetime, timedelta
from lib.utils import log_string


class ODPSDataLoader:
    """
    从 ODPS 加载训练数据的数据加载器
    """
    
    def __init__(self, config, log=None):
        """
        初始化 ODPS 数据加载器
        
        参数:
            config (dict): 配置字典，需包含:
                - odps_project: ODPS 项目名
                - odps_endpoint: ODPS endpoint
                - odps_table: ODPS 表名（默认为 tb_inter_spatial_method_pretrain_data）
                - odps_meta_table: ODPS 元数据表名（可选，包含节点经纬度信息）
                - adcode: 行政区划代码（如 '110000'）
                - start_date: 开始日期 (格式: 'YYYYMMDD')
                - end_date: 结束日期 (格式: 'YYYYMMDD')
                - input_len: 输入序列长度（默认 12）
                - output_len: 输出序列长度（默认 12）
                - train_ratio: 训练集比例（默认 0.7）
                - val_ratio: 验证集比例（默认 0.1）
                - test_ratio: 测试集比例（默认 0.2）
                - recur_times: KD-tree 递归次数（默认 1）
                - spa_patchsize: 空间 patch 大小（默认 4）
            log: 日志文件对象
        """
        self.config = config
        self.log = log
        
        # ODPS 配置
        self.odps_project = config.get('odps_project', 'autonavi_traffic_report')
        self.odps_endpoint = config.get('odps_endpoint', 
                                       'http://service-corp.odps.aliyun-inc.com/api')
        self.odps_table = config.get('odps_table', 'tb_inter_spatial_method_pretrain_data')
        self.odps_meta_table = config.get('odps_meta_table', 'intersection_meta_aligned')  # 默认使用对齐的元数据表
        
        # ⚠️ 元数据表是必须的（用于 KD-tree 空间分组）
        if not self.odps_meta_table:
            raise ValueError(
                "必须提供 odps_meta_table 参数！\n"
                "PatchSTG 需要节点位置信息进行空间 KD-tree 分组。\n"
                "请在配置中添加: odps_meta_table = 'intersection_meta_aligned'"
            )
        
        # 数据过滤条件
        self.adcode = config.get('adcode', None)
        self.start_date = config.get('start_date', None)
        self.end_date = config.get('end_date', None)
        self.limit = config.get('limit', None)  # 可选：限制查询行数（用于测试）
        
        # 训练参数
        self.input_len = config.get('input_len', 12)
        self.output_len = config.get('output_len', 12)
        self.train_ratio = config.get('train_ratio', 0.6)  # 60% 训练集
        self.val_ratio = config.get('val_ratio', 0.2)     # 20% 验证集
        self.test_ratio = config.get('test_ratio', 0.2)   # 20% 测试集
        
        # 空间 patching 参数
        self.recur_times = config.get('recur_times', 1)
        self.spa_patchsize = config.get('spa_patchsize', 4)
        
        # 数据集
        self.trainX = None
        self.trainY = None
        self.trainXTE = None
        self.trainYTE = None
        
        self.valX = None
        self.valY = None
        self.valXTE = None
        self.valYTE = None
        
        self.testX = None
        self.testY = None
        self.testXTE = None
        self.testYTE = None
        
        # 归一化参数
        self.mean = None
        self.std = None
        
        # 节点信息
        self.node_list = None  # [(nds_id, next_nds_id), ...]
        self.node_num = 0
        self.node_to_idx = {}  # {(nds_id, next_nds_id): idx}
        self.node_locations = None  # 节点经纬度 (2, num_nodes): [lat, lng]
        
        # Patch 索引（如果需要空间 patching）
        self.ori_parts_idx = None
        self.reo_parts_idx = None
        self.reo_all_idx = None
        
        self._loaded = False
        self._odps_client = None
    
    def _init_odps_client(self):
        """初始化 ODPS 客户端"""
        if self._odps_client is not None:
            return
        
        access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
        
        if not access_id or not secret:
            raise RuntimeError(
                "缺少 ODPS 凭证。请设置环境变量:\n"
                "  ALIBABA_CLOUD_ACCESS_KEY_ID\n"
                "  ALIBABA_CLOUD_ACCESS_KEY_SECRET"
            )
        
        self._odps_client = ODPS(
            access_id,
            secret,
            project=self.odps_project,
            endpoint=self.odps_endpoint
        )
        
        if self.log:
            log_string(self.log, f'ODPS client initialized for project: {self.odps_project}')
    
    def _build_query(self):
        """构建 SQL 查询语句"""
        where_clauses = []
        
        if self.adcode:
            where_clauses.append(f"adcode = '{self.adcode}'")
        
        if self.start_date and self.end_date:
            where_clauses.append(f"ds >= '{self.start_date}' AND ds <= '{self.end_date}'")
        elif self.start_date:
            where_clauses.append(f"ds >= '{self.start_date}'")
        elif self.end_date:
            where_clauses.append(f"ds <= '{self.end_date}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # 构建 LIMIT 子句
        limit_clause = f"\nLIMIT {self.limit}" if self.limit else ""
        
        query = f"""
        SELECT 
            nds_id,
            next_nds_id,
            adcode,
            ds,
            passts_time,
            flow_label,
            time_feat,
            dym_feat_feat
        FROM {self.odps_table}
        WHERE {where_clause}
        ORDER BY nds_id, next_nds_id, passts_time{limit_clause}
        """
        
        return query
    
    def _parse_time_feat(self, time_feat_str):
        """
        解析时间特征字符串
        
        参数:
            time_feat_str: "week hour minute day_type day month;..." (24段)
            
        返回:
            np.ndarray: shape (24, 6)
        """
        segments = time_feat_str.split(';')
        features = []
        
        for seg in segments:
            parts = seg.strip().split()
            if len(parts) == 6:
                features.append([int(p) for p in parts])
            else:
                # 异常处理：填充默认值
                features.append([0, 0, 0, 0, 0, 0])
        
        # 确保有 24 段
        while len(features) < 24:
            features.append([0, 0, 0, 0, 0, 0])
        
        return np.array(features[:24], dtype=np.float32)
    
    def _parse_dym_feat(self, dym_feat_str):
        """
        解析动态流量特征字符串
        
        参数:
            dym_feat_str: "15;8;0;12;...;3" (24个值)
            
        返回:
            np.ndarray: shape (24,)
        """
        values = dym_feat_str.split(';')
        features = []
        
        for val in values:
            try:
                features.append(float(val.strip()))
            except:
                features.append(0.0)
        
        # 确保有 24 个值
        while len(features) < 24:
            features.append(0.0)
        
        return np.array(features[:24], dtype=np.float32)
    
    def load_data(self):
        """从 ODPS 加载数据"""
        if self._loaded:
            if self.log:
                log_string(self.log, 'Data already loaded, skipping...')
            return
        
        if self.log:
            log_string(self.log, '\n------------ Loading Data from ODPS -------------')
            log_string(self.log, f'Project: {self.odps_project}')
            log_string(self.log, f'Table: {self.odps_table}')
            log_string(self.log, f'Adcode: {self.adcode}')
            log_string(self.log, f'Date range: {self.start_date} ~ {self.end_date}')
        
        # 初始化 ODPS 客户端
        self._init_odps_client()
        
        # 构建并执行查询
        query = self._build_query()
        if self.log:
            log_string(self.log, f'\nExecuting query:\n{query}\n')
        
        # 执行查询并转为 DataFrame
        with self._odps_client.execute_sql(query).open_reader() as reader:
            records = [record.values for record in reader]
            columns = ['nds_id', 'next_nds_id', 'adcode', 'ds', 'passts_time', 
                      'flow_label', 'time_feat', 'dym_feat_feat']
            df = pd.DataFrame(records, columns=columns)
        
        if self.log:
            log_string(self.log, f'Loaded {len(df)} records from ODPS')
        
        if len(df) == 0:
            raise ValueError("No data loaded from ODPS. Check your filter conditions.")
        
        # 构建节点列表
        self._build_node_list(df)
        
        # 加载节点位置信息（如果有元数据表）
        self._load_node_locations()
        
        # 处理数据并划分数据集
        self._process_and_split_data(df)
        
        self._loaded = True
        
        if self.log:
            log_string(self.log, f'Train samples: {self.trainX.shape[0]}')
            log_string(self.log, f'Val samples: {self.valX.shape[0]}')
            log_string(self.log, f'Test samples: {self.testX.shape[0]}')
            log_string(self.log, f'Nodes: {self.node_num}')
            log_string(self.log, f'Mean: {self.mean:.4f}, Std: {self.std:.4f}')
            log_string(self.log, '------------ End -------------\n')
    
    def _build_node_list(self, df):
        """构建节点列表"""
        # 获取唯一的 (nds_id, next_nds_id) 对
        node_pairs = df[['nds_id', 'next_nds_id']].drop_duplicates()
        self.node_list = [(row['nds_id'], row['next_nds_id']) 
                         for _, row in node_pairs.iterrows()]
        self.node_num = len(self.node_list)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        
        if self.log:
            log_string(self.log, f'Found {self.node_num} unique node pairs (road segments)')
    
    def _load_node_locations(self):
        """
        从 ODPS 元数据表加载路口的经纬度信息（必须）
        
        ⚠️ 经纬度是必须的，用于 KD-tree 空间分组
        
        数据关系:
        - (nds_id, next_nds_id) 表示一个转向流
        - 每个转向流对应一个路口 inter_id
        - 路口 inter_id 有经纬度坐标
        
        元数据表应包含以下字段:
        - nds_id: 转向前的路段 ID
        - next_nds_id: 转向后的路段 ID
        - inter_id: 路口 ID
        - lat: 路口纬度
        - lng: 路口经度
        """
        
        if self.log:
            log_string(self.log, f'\n📍 Loading node locations from: {self.odps_meta_table}')
        
        # 构建查询：获取所有转向流对应的路口位置
        query = f"""
        SELECT 
            nds_id,
            next_nds_id,
            inter_id,
            lat,
            lng
        FROM {self.odps_meta_table}
        WHERE 1=1
        """
        
        # 如果有 adcode 过滤，也应用到元数据表
        if self.adcode:
            query += f" AND adcode = '{self.adcode}'"
        
        if self.log:
            log_string(self.log, f'Executing meta query:\n{query}')
        
        # 执行查询
        with self._odps_client.execute_sql(query).open_reader() as reader:
            meta_records = [record.values for record in reader]
            meta_df = pd.DataFrame(meta_records, 
                                  columns=['nds_id', 'next_nds_id', 'inter_id', 
                                         'lat', 'lng'])
        
        if len(meta_df) == 0:
            raise RuntimeError(
                f"❌ 元数据表 {self.odps_meta_table} 中没有找到位置数据！\n"
                f"   过滤条件: adcode = '{self.adcode}'\n"
                f"   PatchSTG 需要节点位置进行空间分组。"
            )
        
        if self.log:
            unique_intersections = meta_df['inter_id'].nunique()
            log_string(self.log, f'   Found {len(meta_df)} turn flows across {unique_intersections} intersections')
        
        # 创建位置数组 (2, num_nodes): [lat, lng]
        self.node_locations = np.zeros((2, self.node_num), dtype=np.float32)
        
        # 填充每个转向流（节点对）的位置（使用对应路口的位置）
        missing_count = 0
        missing_nodes = []
        
        for idx, (nds_id, next_nds_id) in enumerate(self.node_list):
            # 查找对应的路口位置
            location = meta_df[
                (meta_df['nds_id'] == nds_id) & 
                (meta_df['next_nds_id'] == next_nds_id)
            ]
            
            if len(location) > 0:
                lat = location.iloc[0]['lat']
                lng = location.iloc[0]['lng']
                
                # 检查经纬度是否有效
                if lat is None or lng is None or lat == 0 or lng == 0:
                    missing_count += 1
                    missing_nodes.append((nds_id, next_nds_id))
                    self.node_locations[0, idx] = 0.0
                    self.node_locations[1, idx] = 0.0
                else:
                    self.node_locations[0, idx] = lat
                    self.node_locations[1, idx] = lng
            else:
                missing_count += 1
                missing_nodes.append((nds_id, next_nds_id))
                self.node_locations[0, idx] = 0.0
                self.node_locations[1, idx] = 0.0
        
        # 计算覆盖率
        coverage = (self.node_num - missing_count) * 100.0 / self.node_num
        
        if self.log:
            log_string(self.log, f'   ✅ Loaded locations for {self.node_num - missing_count}/{self.node_num} nodes')
            log_string(self.log, f'   📊 Coverage: {coverage:.2f}%')
        
        # ⚠️ 如果覆盖率太低，报错
        if coverage < 50.0:
            raise RuntimeError(
                f"❌ 位置覆盖率太低: {coverage:.2f}%\n"
                f"   只有 {self.node_num - missing_count}/{self.node_num} 个节点有有效位置。\n"
                f"   PatchSTG 需要至少 50% 的节点有位置信息才能进行空间分组。\n"
                f"   请检查元数据表是否包含所有节点的位置信息。"
            )
        
        if missing_count > 0:
            if self.log:
                log_string(self.log, f'   ⚠️  Warning: {missing_count} nodes have missing locations')
                if missing_count <= 10:
                    log_string(self.log, f'   Missing nodes: {missing_nodes[:10]}')
    
    def _process_and_split_data(self, df):
        """
        处理数据并划分为训练/验证/测试集
        
        ✅ 新实现：按时间窗口组织数据（密集格式）
        
        数据结构:
        - X: (num_samples, input_len, num_nodes, 1) - 过去的流量值
        - Y: (num_samples, output_len, num_nodes, 1) - 未来的流量值
        - XTE: (num_samples, input_len, 2) - 时间特征 (tod, dow)
        - YTE: (num_samples, output_len, 2) - 时间特征 (tod, dow)
        
        关键改进：每个样本包含所有节点在连续时间段的数据（密集）
        """
        if self.log:
            log_string(self.log, '\n📊 Processing data (time-series format)...')
        
        # 步骤 1: 转换时间戳格式并添加节点索引
        if self.log:
            log_string(self.log, '   Step 1: Parsing timestamps...')
        
        df['timestamp'] = pd.to_datetime(df['passts_time'])
        df['node_idx'] = df.apply(
            lambda row: self.node_to_idx[(row['nds_id'], row['next_nds_id'])], 
            axis=1
        )
        
        # 步骤 2: 按分钟对齐时间戳（向下取整）
        df['time_minute'] = df['timestamp'].dt.floor('1min')
        
        if self.log:
            time_range = f"{df['time_minute'].min()} ~ {df['time_minute'].max()}"
            log_string(self.log, f'   Time range: {time_range}')
        
        # 步骤 3: Pivot 为时间序列格式（时间 × 节点）
        if self.log:
            log_string(self.log, '   Step 2: Pivoting to time-series format...')
            log_string(self.log, f'   This may take a while for {len(df)} records...')
        
        # 使用 pivot_table 聚合（如果同一节点同一分钟有多条记录，取平均）
        flow_matrix = df.pivot_table(
            index='time_minute',
            columns='node_idx',
            values='flow_label',
            aggfunc='mean',  # 如果有重复，取平均
            fill_value=0.0   # 缺失值填充为 0
        )
        
        # 确保所有节点都在列中（按索引排序）
        all_node_indices = list(range(self.node_num))
        missing_nodes = set(all_node_indices) - set(flow_matrix.columns)
        for node_idx in missing_nodes:
            flow_matrix[node_idx] = 0.0
        flow_matrix = flow_matrix[all_node_indices]  # 按节点索引排序
        
        if self.log:
            log_string(self.log, f'   ✅ Flow matrix shape: {flow_matrix.shape} (time × nodes)')
            log_string(self.log, f'   Time steps: {len(flow_matrix)}')
            log_string(self.log, f'   Nodes: {len(flow_matrix.columns)}')
            
            # 统计非零值比例
            non_zero_ratio = (flow_matrix.values > 0).sum() / flow_matrix.size * 100
            log_string(self.log, f'   Non-zero ratio: {non_zero_ratio:.2f}%')
        
        # 步骤 4: 提取时间特征
        if self.log:
            log_string(self.log, '   Step 3: Extracting time features...')
        
        # 为每个时间点生成时间特征
        time_features = []
        for timestamp in flow_matrix.index:
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
            
            tod = hour / 24.0      # Time of day [0, 1)
            dow = day_of_week / 7.0  # Day of week [0, 1)
            
            time_features.append([tod, dow])
        
        time_features = np.array(time_features, dtype=np.float32)
        
        # 步骤 5: 使用滑动窗口生成样本
        if self.log:
            log_string(self.log, '   Step 4: Generating samples with sliding window...')
        
        flow_values = flow_matrix.values  # (num_times, num_nodes)
        num_times = len(flow_values)
        num_samples = num_times - self.input_len - self.output_len + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough time steps to generate samples!\n"
                f"  Time steps: {num_times}\n"
                f"  Required: input_len ({self.input_len}) + output_len ({self.output_len}) = {self.input_len + self.output_len}\n"
                f"  Please use a longer date range."
            )
        
        if self.log:
            log_string(self.log, f'   Total samples to generate: {num_samples}')
        
        # 预分配数组
        X_data = np.zeros((num_samples, self.input_len, self.node_num, 1), dtype=np.float32)
        Y_data = np.zeros((num_samples, self.output_len, self.node_num, 1), dtype=np.float32)
        XTE_data = np.zeros((num_samples, self.input_len, 2), dtype=np.float32)
        YTE_data = np.zeros((num_samples, self.output_len, 2), dtype=np.float32)
        
        # 滑动窗口生成样本
        for i in range(num_samples):
            # 输入：时间步 i 到 i+input_len-1
            X_data[i, :, :, 0] = flow_values[i:i+self.input_len]
            XTE_data[i] = time_features[i:i+self.input_len]
            
            # 输出：时间步 i+input_len 到 i+input_len+output_len-1
            Y_data[i, :, :, 0] = flow_values[i+self.input_len:i+self.input_len+self.output_len]
            YTE_data[i] = time_features[i+self.input_len:i+self.input_len+self.output_len]
        
        if self.log:
            log_string(self.log, f'   ✅ Generated {num_samples} samples')
            log_string(self.log, f'   X shape: {X_data.shape}')
            log_string(self.log, f'   Y shape: {Y_data.shape}')
        
        # 步骤 6: 验证数据有效性
        if self.log:
            log_string(self.log, '   Step 5: Validating data...')
        
        # 检查是否有无效值
        if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
            raise ValueError("X_data contains NaN or Inf values!")
        if np.any(np.isnan(Y_data)) or np.any(np.isinf(Y_data)):
            raise ValueError("Y_data contains NaN or Inf values!")
        
        # 统计每个样本中有多少节点有非零值
        nodes_per_sample = (X_data[:, :, :, 0] > 0).any(axis=1).sum(axis=1)
        avg_nodes = nodes_per_sample.mean()
        min_nodes = nodes_per_sample.min()
        max_nodes = nodes_per_sample.max()
        
        if self.log:
            log_string(self.log, f'   Nodes with data per sample: min={min_nodes}, max={max_nodes}, avg={avg_nodes:.1f}')
            log_string(self.log, f'   Flow value range: [{X_data.min():.2f}, {X_data.max():.2f}]')
        
        # 步骤 7: 计算归一化参数（基于训练集）
        num_train = int(num_samples * self.train_ratio)
        train_data = X_data[:num_train]
        
        # 只对非零值计算均值和标准差（更准确）
        train_nonzero = train_data[train_data > 0]
        if len(train_nonzero) > 0:
            self.mean = np.mean(train_nonzero)
            self.std = np.std(train_nonzero)
        else:
            self.mean = 0.0
            self.std = 1.0
        
        if self.std < 1e-6:
            self.std = 1.0
            if self.log:
                log_string(self.log, '   ⚠️  Warning: std is too small, set to 1.0')
        
        if self.log:
            log_string(self.log, f'   Normalization: mean={self.mean:.4f}, std={self.std:.4f}')
        
        # 步骤 8: 划分数据集
        num_val = int(num_samples * self.val_ratio)
        num_test = int(num_samples * self.test_ratio)
        
        # 确保划分后至少有一些样本
        if num_train < 1 or num_val < 1 or num_test < 1:
            raise ValueError(
                f"Dataset too small after split!\n"
                f"  Total samples: {num_samples}\n"
                f"  Train: {num_train}, Val: {num_val}, Test: {num_test}\n"
                f"  Please use a longer date range or adjust split ratios."
            )
        
        self.trainX = X_data[:num_train]
        self.trainY = Y_data[:num_train]
        self.trainXTE = XTE_data[:num_train]
        self.trainYTE = YTE_data[:num_train]
        
        self.valX = X_data[num_train:num_train+num_val]
        self.valY = Y_data[num_train:num_train+num_val]
        self.valXTE = XTE_data[num_train:num_train+num_val]
        self.valYTE = YTE_data[num_train:num_train+num_val]
        
        self.testX = X_data[num_train+num_val:num_train+num_val+num_test]
        self.testY = Y_data[num_train+num_val:num_train+num_val+num_test]
        self.testXTE = XTE_data[num_train+num_val:num_train+num_val+num_test]
        self.testYTE = YTE_data[num_train+num_val:num_train+num_val+num_test]
        
        if self.log:
            log_string(self.log, f'   ✅ Dataset split: Train={num_train}, Val={num_val}, Test={num_test}')
        
        # 创建空间 patch 索引
        self._create_spatial_patches(self.trainX)
    
    def _create_spatial_patches(self, train_data):
        """
        创建空间 patch 索引（使用 KD-tree）
        
        ⚠️ 必须有节点位置信息才能进行空间分组
        """
        if self.log:
            log_string(self.log, '\n🌳 Creating spatial patches using KD-tree...')
        
        # 检查是否有位置信息
        if self.node_locations is None:
            raise RuntimeError(
                "❌ 没有节点位置信息，无法创建空间 patch！\n"
                "   请确保已加载元数据表。"
            )
        
        # 检查位置信息是否有效
        valid_locations = (self.node_locations != 0).any(axis=0).sum()
        if valid_locations == 0:
            raise RuntimeError(
                "❌ 所有节点的位置信息都无效（全为0）！\n"
                "   请检查元数据表中的 lat, lng 字段。"
            )
        
        try:
            # 导入原有的 patching 函数
            from lib.utils import construct_adj, reorderData
            from sklearn.neighbors import KDTree
            
            if self.log:
                log_string(self.log, f'   Node locations shape: {self.node_locations.shape}')
                log_string(self.log, f'   Valid locations: {valid_locations}/{self.node_num}')
                log_string(self.log, f'   Lat range: [{self.node_locations[0].min():.4f}, {self.node_locations[0].max():.4f}]')
                log_string(self.log, f'   Lng range: [{self.node_locations[1].min():.4f}, {self.node_locations[1].max():.4f}]')
            
            # 使用 KD-tree 进行空间划分
            tree = KDTree(self.node_locations.T)  # (num_nodes, 2)
            
            # 递归划分
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
            
            parts_idx = recursive_split(np.arange(self.node_num))
            
            if self.log:
                log_string(self.log, f'   ✅ Created {len(parts_idx)} spatial patches')
                patch_sizes = [len(p) for p in parts_idx]
                log_string(self.log, f'   Patch size range: [{min(patch_sizes)}, {max(patch_sizes)}]')
                log_string(self.log, f'   Average patch size: {np.mean(patch_sizes):.1f}')
            
            # 构造邻接矩阵用于补齐
            if self.log:
                log_string(self.log, '   Constructing adjacency matrix...')
            # construct_adj 期望 (time_steps, nodes, 1)，但 train_data 是 (samples, time_steps, nodes, 1)
            # 我们将所有样本拼接成一个长时间序列
            train_data_concat = train_data.reshape(-1, self.node_num, 1)  # (samples*time_steps, nodes, 1)
            adj = construct_adj(train_data_concat, self.node_num)
            
            # 获取最大 patch 长度
            mxlen = max([len(p) for p in parts_idx])
            
            # 重排并补齐
            if self.log:
                log_string(self.log, '   Reordering patches...')
            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = reorderData(
                parts_idx, mxlen, adj, self.spa_patchsize
            )
            
            if self.log:
                log_string(self.log, '   ✅ Spatial patching completed')
            
        except ImportError as e:
            raise RuntimeError(
                f"❌ 缺少必要的库: {str(e)}\n"
                f"   请安装: pip install scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ 空间 patching 失败: {str(e)}\n"
                f"   这是创建 KD-tree 空间分组时的错误。"
            )
    
    def get_train_data(self):
        """获取训练数据"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.trainX, self.trainY, self.trainXTE, self.trainYTE
    
    def get_val_data(self):
        """获取验证数据"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.valX, self.valY, self.valXTE, self.valYTE
    
    def get_test_data(self):
        """获取测试数据"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.testX, self.testY, self.testXTE, self.testYTE
    
    def get_normalization_params(self):
        """获取归一化参数"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.mean, self.std
    
    def get_patch_indices(self):
        """获取 patch 索引"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx
    
    def shuffle_train_data(self, seed=None):
        """打乱训练数据"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        if seed is not None:
            np.random.seed(seed)
        
        num_train = self.trainX.shape[0]
        permutation = np.random.permutation(num_train)
        
        self.trainX = self.trainX[permutation]
        self.trainY = self.trainY[permutation]
        self.trainXTE = self.trainXTE[permutation]
        
        if self.log:
            log_string(self.log, 'Training data shuffled')
    
    def normalize_data(self, data):
        """归一化数据"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return (data - self.mean) / self.std
    
    def denormalize_data(self, data):
        """反归一化数据"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return data * self.std + self.mean
    
    def get_data_info(self):
        """获取数据集信息"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        return {
            'train_samples': self.trainX.shape[0],
            'val_samples': self.valX.shape[0],
            'test_samples': self.testX.shape[0],
            'input_shape': self.trainX.shape[1:],
            'output_shape': self.trainY.shape[1:],
            'mean': float(self.mean),
            'std': float(self.std),
            'num_nodes': self.node_num,
            'node_list': self.node_list[:10] if len(self.node_list) > 10 else self.node_list,
            'adcode': self.adcode,
            'date_range': f'{self.start_date} ~ {self.end_date}'
        }
