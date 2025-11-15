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
        """
        从 ODPS 加载数据（流式读取版本）
        
        ✅ 改进：使用 Table Iterator 流式读取，避免内存溢出
        """
        if self._loaded:
            if self.log:
                log_string(self.log, 'Data already loaded, skipping...')
            return
        
        if self.log:
            log_string(self.log, '\n------------ Loading Data from ODPS (Streaming) -------------')
            log_string(self.log, f'Project: {self.odps_project}')
            log_string(self.log, f'Table: {self.odps_table}')
            log_string(self.log, f'Adcode: {self.adcode}')
            log_string(self.log, f'Date range: {self.start_date} ~ {self.end_date}')
        
        # 初始化 ODPS 客户端
        self._init_odps_client()
        
        # 步骤 1: 先查询节点列表（用小查询获取唯一节点对）
        if self.log:
            log_string(self.log, '\nStep 1: Loading node list...')
        self._load_node_list_from_odps()
        
        # 步骤 2: 加载节点位置信息
        if self.log:
            log_string(self.log, '\nStep 2: Loading node locations...')
        self._load_node_locations()
        
        # 步骤 3: 流式读取数据并处理
        if self.log:
            log_string(self.log, '\nStep 3: Streaming data from ODPS...')
        self._stream_and_process_data()
        
        self._loaded = True
        
        if self.log:
            log_string(self.log, f'\n✅ Data loading completed!')
            log_string(self.log, f'Train samples: {self.trainX.shape[0]}')
            log_string(self.log, f'Val samples: {self.valX.shape[0]}')
            log_string(self.log, f'Test samples: {self.testX.shape[0]}')
            log_string(self.log, f'Nodes: {self.node_num}')
            log_string(self.log, f'Mean: {self.mean:.4f}, Std: {self.std:.4f}')
            log_string(self.log, '------------ End -------------\n')
    
    def load_data_for_date_range(self, start_date, end_date):
        """
        为指定日期范围加载数据（方案 3：分批加载训练）
        
        📌 用法：在训练循环中多次调用，每次加载不同日期的数据
        
        参数:
            start_date (str): 开始日期 'YYYYMMDD'
            end_date (str): 结束日期 'YYYYMMDD'
        
        示例:
            # 每次训练加载 2 天数据
            for date_batch in date_chunks:
                data_loader.load_data_for_date_range('20250919', '20250920')
                trainX, trainY, trainXTE, trainYTE = data_loader.get_train_data()
                # 训练这批数据...
                data_loader.clear_data()  # 释放内存
        """
        if self.log:
            log_string(self.log, f'\n🔄 Loading data for date range: {start_date} ~ {end_date}')
        
        # 临时修改配置的日期范围
        original_start = self.start_date
        original_end = self.end_date
        self.start_date = start_date
        self.end_date = end_date
        
        # 如果是首次加载，需要初始化客户端和节点列表
        if self._odps_client is None:
            self._init_odps_client()
        
        if self.node_list is None:
            if self.log:
                log_string(self.log, 'Step 1: Loading node list (first time)...')
            # 使用原始完整日期范围获取节点列表
            self.start_date = original_start
            self.end_date = original_end
            self._load_node_list_from_odps()
            # 恢复当前批次的日期范围
            self.start_date = start_date
            self.end_date = end_date
        
        if self.node_locations is None:
            if self.log:
                log_string(self.log, 'Step 2: Loading node locations (first time)...')
            self._load_node_locations()
        
        # 流式读取当前日期范围的数据
        if self.log:
            log_string(self.log, 'Step 3: Streaming data for this date range...')
        self._stream_and_process_data()
        
        # 恢复原始日期配置
        self.start_date = original_start
        self.end_date = original_end
        self._loaded = True
        
        if self.log:
            log_string(self.log, f'✅ Loaded {self.trainX.shape[0]} samples for {start_date} ~ {end_date}\n')
    
    def clear_data(self):
        """
        清空已加载的数据，释放内存
        
        📌 用于分批加载场景：训练完当前批次后释放内存
        """
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
        self._loaded = False
        
        if self.log:
            log_string(self.log, '🗑️  Data cleared, memory released')
    
    def _load_node_list_from_odps(self):
        """
        从 ODPS 查询唯一的节点列表
        
        使用 DISTINCT 查询，数据量小，不会有内存问题
        """
        query = f"""
        SELECT DISTINCT 
            nds_id,
            next_nds_id
        FROM {self.odps_table}
        WHERE 1=1
        """
        
        if self.adcode:
            query += f" AND adcode = '{self.adcode}'"
        
        if self.start_date and self.end_date:
            query += f" AND ds >= '{self.start_date}' AND ds <= '{self.end_date}'"
        elif self.start_date:
            query += f" AND ds >= '{self.start_date}'"
        elif self.end_date:
            query += f" AND ds <= '{self.end_date}'"
        
        if self.log:
            log_string(self.log, f'   Querying unique nodes...')
        
        # 执行查询
        with self._odps_client.execute_sql(query).open_reader() as reader:
            node_pairs = [(record[0], record[1]) for record in reader]
        
        self.node_list = node_pairs
        self.node_num = len(self.node_list)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        
        if self.log:
            log_string(self.log, f'   ✅ Found {self.node_num} unique node pairs')
    
    def _stream_and_process_data(self):
        """
        流式读取 ODPS 数据并处理
        
        ✅ 核心改进：使用 Table API 直接读取，支持分片和流式处理
        """
        # 构建查询（用于获取表）
        query = self._build_query()
        
        if self.log:
            log_string(self.log, f'   Executing streaming query...')
            log_string(self.log, f'   Query:\n{query}')
        
        # 方案：使用 execute_sql 的 open_reader 但分批读取
        # open_reader 返回一个迭代器，我们可以分批处理
        
        chunk_size = 100000  # 每批处理 10 万条记录
        total_records = 0
        
        # 用于累积时间序列数据的字典
        # key: time_minute, value: {node_idx: flow_value}
        time_series_dict = {}
        
        if self.log:
            log_string(self.log, f'   Reading data in chunks of {chunk_size} records...')
        
        with self._odps_client.execute_sql(query).open_reader() as reader:
            chunk_records = []
            
            for record in reader:
                chunk_records.append(record.values)
                
                # 达到批次大小，处理这批数据
                if len(chunk_records) >= chunk_size:
                    self._process_chunk(chunk_records, time_series_dict)
                    total_records += len(chunk_records)
                    
                    if self.log:
                        log_string(self.log, f'   Processed {total_records} records...')
                    
                    chunk_records = []
            
            # 处理最后一批
            if chunk_records:
                self._process_chunk(chunk_records, time_series_dict)
                total_records += len(chunk_records)
        
        if self.log:
            log_string(self.log, f'   ✅ Total records processed: {total_records}')
            log_string(self.log, f'   Unique time steps: {len(time_series_dict)}')
        
        if total_records == 0:
            raise ValueError("No data loaded from ODPS. Check your filter conditions.")
        
        # 转换为 DataFrame 并继续后续处理
        if self.log:
            log_string(self.log, '   Converting to time series format...')
        
        self._build_time_series_from_dict(time_series_dict)
    
    def _process_chunk(self, records, time_series_dict):
        """
        处理一批记录，累积到时间序列字典中
        
        参数:
            records: 记录列表
            time_series_dict: 累积的时间序列字典
        """
        columns = ['nds_id', 'next_nds_id', 'adcode', 'ds', 'passts_time', 
                  'flow_label', 'time_feat', 'dym_feat_feat']
        df_chunk = pd.DataFrame(records, columns=columns)
        
        # 转换时间戳
        df_chunk['timestamp'] = pd.to_datetime(df_chunk['passts_time'])
        df_chunk['time_minute'] = df_chunk['timestamp'].dt.floor('1min')
        
        # 添加节点索引
        df_chunk['node_idx'] = df_chunk.apply(
            lambda row: self.node_to_idx.get((row['nds_id'], row['next_nds_id']), -1), 
            axis=1
        )
        
        # 过滤掉未知节点
        df_chunk = df_chunk[df_chunk['node_idx'] != -1]
        
        # 累积到字典中
        for _, row in df_chunk.iterrows():
            time_key = row['time_minute']
            node_idx = row['node_idx']
            flow_value = row['flow_label']
            
            if time_key not in time_series_dict:
                time_series_dict[time_key] = {}
            
            # 如果同一节点同一时间有多条记录，取平均
            if node_idx in time_series_dict[time_key]:
                time_series_dict[time_key][node_idx] = (
                    time_series_dict[time_key][node_idx] + flow_value
                ) / 2
            else:
                time_series_dict[time_key][node_idx] = flow_value
    
    def _build_time_series_from_dict(self, time_series_dict):
        """
        从时间序列字典构建最终的训练数据
        
        参数:
            time_series_dict: {time_minute: {node_idx: flow_value}}
        """
        # 排序时间点
        sorted_times = sorted(time_series_dict.keys())
        num_times = len(sorted_times)
        
        if self.log:
            log_string(self.log, f'   Time range: {sorted_times[0]} ~ {sorted_times[-1]}')
            log_string(self.log, f'   Time steps: {num_times}')
        
        # 构建流量矩阵 (num_times, num_nodes)
        flow_matrix = np.zeros((num_times, self.node_num), dtype=np.float32)
        
        for t_idx, time_key in enumerate(sorted_times):
            node_flows = time_series_dict[time_key]
            for node_idx, flow_value in node_flows.items():
                flow_matrix[t_idx, node_idx] = flow_value
        
        if self.log:
            non_zero_ratio = (flow_matrix > 0).sum() / flow_matrix.size * 100
            log_string(self.log, f'   Flow matrix shape: {flow_matrix.shape}')
            log_string(self.log, f'   Non-zero ratio: {non_zero_ratio:.2f}%')
        
        # 构建时间特征
        time_features = []
        for timestamp in sorted_times:
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            tod = hour / 24.0
            dow = day_of_week / 7.0
            time_features.append([tod, dow])
        
        time_features = np.array(time_features, dtype=np.float32)
        
        # 生成样本
        if self.log:
            log_string(self.log, '   Generating samples with sliding window...')
        
        num_samples = num_times - self.input_len - self.output_len + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough time steps to generate samples!\n"
                f"  Time steps: {num_times}\n"
                f"  Required: input_len ({self.input_len}) + output_len ({self.output_len}) = {self.input_len + self.output_len}\n"
                f"  Please use a longer date range."
            )
        
        # 预分配数组
        X_data = np.zeros((num_samples, self.input_len, self.node_num, 1), dtype=np.float32)
        Y_data = np.zeros((num_samples, self.output_len, self.node_num, 1), dtype=np.float32)
        XTE_data = np.zeros((num_samples, self.input_len, 2), dtype=np.float32)
        YTE_data = np.zeros((num_samples, self.output_len, 2), dtype=np.float32)
        
        # 滑动窗口生成样本
        for i in range(num_samples):
            X_data[i, :, :, 0] = flow_matrix[i:i+self.input_len]
            XTE_data[i] = time_features[i:i+self.input_len]
            Y_data[i, :, :, 0] = flow_matrix[i+self.input_len:i+self.input_len+self.output_len]
            YTE_data[i] = time_features[i+self.input_len:i+self.input_len+self.output_len]
        
        if self.log:
            log_string(self.log, f'   ✅ Generated {num_samples} samples')
        
        # 验证数据
        if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
            raise ValueError("X_data contains NaN or Inf values!")
        if np.any(np.isnan(Y_data)) or np.any(np.isinf(Y_data)):
            raise ValueError("Y_data contains NaN or Inf values!")
        
        # 计算归一化参数
        num_train = int(num_samples * self.train_ratio)
        train_data = X_data[:num_train]
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
            log_string(self.log, f'   Normalization: mean={self.mean:.4f}, std={self.std:.4f}')
        
        # 划分数据集
        num_val = int(num_samples * self.val_ratio)
        num_test = int(num_samples * self.test_ratio)
        
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
        
        # 创建空间 patch
        self._create_spatial_patches(self.trainX)

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
