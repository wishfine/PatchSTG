"""
使用 OdpsTableDataset 直接读取 ODPS 表的数据加载器
支持关联元数据表获取路口经纬度信息
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.utils import log_string

try:
    from utils.odps_table import OdpsTableDataset
except ImportError:
    print("Warning: OdpsTableDataset not found, using fallback implementation")
    from odps import ODPS
    
    class OdpsTableDataset:
        """简单的 ODPS 表读取实现（如果没有专用工具）"""
        def __init__(self, table_path, slice_id=0, slice_count=1):
            from odps import ODPS
            
            access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
            secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
            
            if not access_id or not secret:
                raise RuntimeError("缺少 ODPS 凭证环境变量")
            
            # 解析表路径: odps://project/tables/table_name
            parts = table_path.replace('odps://', '').split('/tables/')
            project = parts[0]
            table_name = parts[1]
            
            self.odps = ODPS(
                access_id, 
                secret, 
                project=project,
                endpoint='http://service-corp.odps.aliyun-inc.com/api'
            )
            self.table_name = table_name
            self.slice_id = slice_id
            self.slice_count = slice_count
            self._data = []
            self._load_data()
        
        def _load_data(self):
            """从 ODPS 表加载数据"""
            # 简单实现：直接查询（实际应该支持分片和增量读取）
            query = f"SELECT * FROM {self.table_name} LIMIT 10000"
            print(f"Loading data from {self.table_name}...")
            
            with self.odps.execute_sql(query).open_reader() as reader:
                for record in reader:
                    self._data.append(list(record.values))
            
            print(f"Loaded {len(self._data)} records")
        
        def __len__(self):
            return len(self._data)
        
        def __getitem__(self, idx):
            return self._data[idx]


class ODPSTableDataLoader:
    """
    使用 OdpsTableDataset 直接读取 ODPS 表的数据加载器
    
    特点:
    - 直接读取 ODPS 表，不需要写 SQL
    - 支持加载元数据表获取经纬度
    - 支持分布式训练
    """
    
    def __init__(self, config, log=None):
        """
        初始化数据加载器
        
        参数:
            config (dict): 配置字典，需包含:
                - odps_project: ODPS 项目名
                - odps_table: ODPS 主表名
                - odps_meta_table: ODPS 元数据表名（可选）
                - adcode: 城市代码
                - start_date: 开始日期
                - end_date: 结束日期
                - batch_size: 批大小
                - num_workers: 工作进程数
                - input_len: 输入序列长度
                - output_len: 输出序列长度（如果需要）
        """
        self.config = config
        self.log = log
        
        # ODPS 配置
        self.odps_project = config.get('odps_project', 'autonavi_traffic_report')
        self.odps_table = config.get('odps_table', 'tb_inter_spatial_method_pretrain_data')
        self.odps_meta_table = config.get('odps_meta_table', 'intersection_meta_1')
        
        # 数据过滤
        self.adcode = config.get('adcode', None)
        self.start_date = config.get('start_date', None)
        self.end_date = config.get('end_date', None)
        
        # DataLoader 参数
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('num_workers', 4)
        self.input_len = config.get('input_len', 12)
        
        # 分布式训练参数
        self.slice_id = int(os.environ.get('RANK', 0))
        self.slice_count = int(os.environ.get('WORLD_SIZE', 1))
        
        # 位置信息
        self.location_dict = None
        self.use_location = self.odps_meta_table is not None
        
        # 节点信息
        self.node_to_idx = {}
        self.node_list = []
        self.node_num = 0
        self.node_locations = None
        
        if self.log:
            log_string(self.log, f'ODPSTableDataLoader initialized')
            log_string(self.log, f'Project: {self.odps_project}')
            log_string(self.log, f'Table: {self.odps_table}')
            log_string(self.log, f'Meta table: {self.odps_meta_table}')
    
    def _load_location_dict(self):
        """从元数据表加载位置信息字典"""
        if not self.use_location:
            if self.log:
                log_string(self.log, '[SKIP] No meta table specified')
            return
        
        if self.log:
            log_string(self.log, f'\n[STEP 1] Loading location data from {self.odps_meta_table}...')
        
        from odps import ODPS
        
        access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
        
        # 断言：环境变量必须存在
        assert access_id is not None, "❌ ALIBABA_CLOUD_ACCESS_KEY_ID not found in environment"
        assert secret is not None, "❌ ALIBABA_CLOUD_ACCESS_KEY_SECRET not found in environment"
        
        odps = ODPS(
            access_id,
            secret,
            project=self.odps_project,
            endpoint='http://service-corp.odps.aliyun-inc.com/api'
        )
        
        # 构建查询
        query = f"""
        SELECT nds_id, next_nds_id, inter_id, lat, lng, adcode
        FROM {self.odps_project}.{self.odps_meta_table}
        WHERE 1=1
        """
        
        if self.adcode:
            query += f" AND adcode = '{self.adcode}'"
        
        if self.log:
            log_string(self.log, f'Query: {query}')
        
        self.location_dict = {}
        
        try:
            with odps.execute_sql(query).open_reader() as reader:
                for record in reader:
                    nds_id = record[0]
                    next_nds_id = record[1]
                    inter_id = record[2]
                    lat = record[3]
                    lng = record[4]
                    
                    # 断言：位置数据必须有效
                    assert lat is not None and lng is not None, \
                        f"❌ Invalid location for ({nds_id}, {next_nds_id}): lat={lat}, lng={lng}"
                    assert -90 <= lat <= 90, f"❌ Invalid latitude: {lat}"
                    assert -180 <= lng <= 180, f"❌ Invalid longitude: {lng}"
                    
                    key = (nds_id, next_nds_id)
                    self.location_dict[key] = {
                        'inter_id': inter_id,
                        'lat': lat,
                        'lng': lng
                    }
            
            # 断言：必须加载到位置数据
            assert len(self.location_dict) > 0, \
                f"❌ No location data loaded from {self.odps_meta_table}"
            
            if self.log:
                log_string(self.log, f'✅ Loaded {len(self.location_dict)} location records')
        
        except Exception as e:
            if self.log:
                log_string(self.log, f'❌ Error loading location data: {e}')
            self.use_location = False
            raise
    
    def _build_node_list(self, batch_data):
        """从第一批数据构建节点列表"""
        if self.log:
            log_string(self.log, f'\n[STEP 2] Building node list from first batch...')
        
        node_set = set()
        
        for record in batch_data:
            nds_id = record[0]
            next_nds_id = record[1]
            
            # 断言：ID 必须有效
            assert nds_id is not None and next_nds_id is not None, \
                f"❌ Invalid node IDs: nds_id={nds_id}, next_nds_id={next_nds_id}"
            
            node_set.add((nds_id, next_nds_id))
        
        # 断言：必须有节点
        assert len(node_set) > 0, "❌ No nodes found in first batch"
        
        self.node_list = sorted(list(node_set))
        self.node_num = len(self.node_list)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        
        if self.log:
            log_string(self.log, f'✅ Found {self.node_num} unique nodes (turn flows)')
            log_string(self.log, f'   First 5 nodes: {self.node_list[:5]}')
        
        # 构建位置数组
        if self.use_location and self.location_dict:
            self._build_location_array()
        else:
            if self.log:
                log_string(self.log, '⚠️  No location data available, spatial patching disabled')
    
    def _build_location_array(self):
        """构建节点位置数组"""
        if not self.location_dict:
            if self.log:
                log_string(self.log, '⚠️  location_dict is empty')
            return
        
        if self.log:
            log_string(self.log, f'\n[STEP 3] Building location array for {self.node_num} nodes...')
        
        # 断言：必须先有节点列表
        assert self.node_num > 0, "❌ node_num must be > 0"
        assert len(self.node_list) == self.node_num, \
            f"❌ node_list length mismatch: {len(self.node_list)} != {self.node_num}"
        
        self.node_locations = np.zeros((2, self.node_num), dtype=np.float32)
        
        missing_count = 0
        missing_nodes = []
        
        for idx, (nds_id, next_nds_id) in enumerate(self.node_list):
            key = (nds_id, next_nds_id)
            
            if key in self.location_dict:
                loc = self.location_dict[key]
                self.node_locations[0, idx] = loc['lat']
                self.node_locations[1, idx] = loc['lng']
                
                # 断言：位置必须有效
                assert not np.isnan(self.node_locations[0, idx]), \
                    f"❌ NaN latitude for node {idx}: {key}"
                assert not np.isnan(self.node_locations[1, idx]), \
                    f"❌ NaN longitude for node {idx}: {key}"
            else:
                missing_count += 1
                self.node_locations[0, idx] = 0.0
                self.node_locations[1, idx] = 0.0
                
                if len(missing_nodes) < 5:  # 只记录前5个
                    missing_nodes.append(key)
        
        # 统计和断言
        coverage = (self.node_num - missing_count) / self.node_num * 100
        
        if self.log:
            log_string(self.log, f'✅ Location coverage: {coverage:.2f}% ({self.node_num - missing_count}/{self.node_num})')
            
            if missing_count > 0:
                log_string(self.log, f'⚠️  {missing_count} nodes missing location data')
                log_string(self.log, f'   Example missing nodes: {missing_nodes[:5]}')
            
            # 位置范围统计
            lat_min, lat_max = self.node_locations[0, :].min(), self.node_locations[0, :].max()
            lng_min, lng_max = self.node_locations[1, :].min(), self.node_locations[1, :].max()
            log_string(self.log, f'   Latitude range: [{lat_min:.4f}, {lat_max:.4f}]')
            log_string(self.log, f'   Longitude range: [{lng_min:.4f}, {lng_max:.4f}]')
        
        # 断言：不能所有位置都是0
        assert not (self.node_locations.sum() == 0), \
            "❌ All node locations are zero! Check location_dict coverage."
        
        # 警告：如果覆盖率太低
        if coverage < 50:
            warning_msg = f"⚠️  WARNING: Location coverage is only {coverage:.1f}%, spatial patching may not work well"
            if self.log:
                log_string(self.log, warning_msg)
            print(warning_msg)
    
    def _collate_fn(self, batch):
        """
        批处理函数
        
        输入: batch 是列表，每个元素是一条记录
            [nds_id, next_nds_id, adcode, ds, passts_time, flow_label, time_feat, dym_feat_feat]
        
        输出:
            - X: (batch_size, input_len, num_nodes, 1) - 输入流量
            - Y: (batch_size, 1, num_nodes, 1) - 输出流量（当前时刻）
            - TE: (batch_size, input_len, 2) - 时间特征 [tod, dow]
        """
        
        # 如果还没有节点列表，先构建
        if not self.node_list:
            if self.log:
                log_string(self.log, '\n' + '='*60)
                log_string(self.log, 'Processing FIRST BATCH - Building node list')
                log_string(self.log, '='*60)
            self._build_node_list(batch)
        
        batch_size = len(batch)
        
        # 断言：batch 不能为空
        assert batch_size > 0, "❌ Empty batch"
        
        if self.log and not hasattr(self, '_logged_batch_info'):
            log_string(self.log, f'\n[STEP 4] Processing batch data...')
            log_string(self.log, f'   Batch size: {batch_size}')
            log_string(self.log, f'   Node num: {self.node_num}')
            log_string(self.log, f'   Input len: {self.input_len}')
            self._logged_batch_info = True
        
        # 初始化数据数组
        X = np.zeros((batch_size, self.input_len, self.node_num, 1), dtype=np.float32)
        Y = np.zeros((batch_size, 1, self.node_num, 1), dtype=np.float32)
        TE = np.zeros((batch_size, self.input_len, 2), dtype=np.float32)
        
        parse_errors = []
        missing_nodes = []
        
        for i, record in enumerate(batch):
            try:
                # 断言：记录长度必须正确
                assert len(record) >= 8, \
                    f"❌ Record {i} has wrong length: {len(record)}, expected >= 8"
                
                nds_id = record[0]
                next_nds_id = record[1]
                flow_label = record[5]
                time_feat_str = record[6]
                dym_feat_str = record[7]
                
                # 获取节点索引
                node_key = (nds_id, next_nds_id)
                if node_key not in self.node_to_idx:
                    missing_nodes.append(node_key)
                    continue  # 跳过这条记录
                
                node_idx = self.node_to_idx[node_key]
                
                # 断言：索引必须有效
                assert 0 <= node_idx < self.node_num, \
                    f"❌ Invalid node_idx: {node_idx}, node_num: {self.node_num}"
                
                # ========== 解析时间特征 ==========
                try:
                    time_segments = time_feat_str.split(';')
                    
                    # 断言：必须有足够的时间段
                    assert len(time_segments) >= self.input_len, \
                        f"❌ time_feat has {len(time_segments)} segments, need >= {self.input_len}"
                    
                    time_array = np.array([
                        [int(x) for x in seg.split(' ')]
                        for seg in time_segments[:self.input_len]
                    ], dtype=np.int32)
                    
                    # 断言：时间特征维度正确
                    assert time_array.shape == (self.input_len, 6), \
                        f"❌ time_array shape wrong: {time_array.shape}, expected ({self.input_len}, 6)"
                    
                    # 提取 tod 和 dow
                    tod = time_array[:, 1] / 24.0  # hour / 24
                    dow = time_array[:, 0] / 7.0   # week / 7
                    
                    # 断言：tod 和 dow 范围正确
                    assert np.all((tod >= 0) & (tod <= 1)), \
                        f"❌ tod out of range: [{tod.min()}, {tod.max()}]"
                    assert np.all((dow >= 0) & (dow <= 1)), \
                        f"❌ dow out of range: [{dow.min()}, {dow.max()}]"
                    
                    TE[i, :, 0] = tod
                    TE[i, :, 1] = dow
                    
                except Exception as e:
                    parse_errors.append(f"Record {i} time_feat parse error: {e}")
                    pass
                
                # ========== 解析历史流量 ==========
                try:
                    flow_segments = dym_feat_str.split(';')
                    
                    # 断言：必须有足够的流量值
                    assert len(flow_segments) >= self.input_len, \
                        f"❌ dym_feat has {len(flow_segments)} values, need >= {self.input_len}"
                    
                    flow_array = np.array([
                        float(x) for x in flow_segments[:self.input_len]
                    ], dtype=np.float32)
                    
                    # 断言：流量值合理
                    assert not np.any(np.isnan(flow_array)), \
                        f"❌ flow_array contains NaN"
                    assert not np.any(np.isinf(flow_array)), \
                        f"❌ flow_array contains Inf"
                    
                    # 填充到 X（只填充当前节点的位置）
                    X[i, :, node_idx, 0] = flow_array
                    
                except Exception as e:
                    parse_errors.append(f"Record {i} dym_feat parse error: {e}")
                    pass
                
                # 当前时刻流量作为标签
                Y[i, 0, node_idx, 0] = float(flow_label)
                
            except Exception as e:
                parse_errors.append(f"Record {i} general error: {e}")
                continue
        
        # ========== 统计和警告 ==========
        if len(parse_errors) > 0 and self.log:
            log_string(self.log, f'⚠️  {len(parse_errors)} records had parse errors')
            for err in parse_errors[:5]:  # 只显示前5个
                log_string(self.log, f'   {err}')
        
        if len(missing_nodes) > 0 and self.log:
            log_string(self.log, f'⚠️  {len(missing_nodes)} records have nodes not in node_list')
        
        # ========== 数据质量检查 ==========
        X_nonzero = np.count_nonzero(X)
        Y_nonzero = np.count_nonzero(Y)
        X_total = X.size
        Y_total = Y.size
        
        if self.log and not hasattr(self, '_logged_sparsity'):
            log_string(self.log, f'\n[DATA QUALITY CHECK]')
            log_string(self.log, f'   X sparsity: {X_nonzero}/{X_total} ({X_nonzero/X_total*100:.2f}%)')
            log_string(self.log, f'   Y sparsity: {Y_nonzero}/{Y_total} ({Y_nonzero/Y_total*100:.2f}%)')
            log_string(self.log, f'   X range: [{X.min():.2f}, {X.max():.2f}]')
            log_string(self.log, f'   Y range: [{Y.min():.2f}, {Y.max():.2f}]')
            log_string(self.log, f'   TE range: [{TE.min():.4f}, {TE.max():.4f}]')
            self._logged_sparsity = True
        
        # 断言：数据不能全是0
        assert X_nonzero > 0, "❌ X is all zeros!"
        assert Y_nonzero > 0, "❌ Y is all zeros!"
        
        # 断言：形状正确
        assert X.shape == (batch_size, self.input_len, self.node_num, 1), \
            f"❌ X shape wrong: {X.shape}"
        assert Y.shape == (batch_size, 1, self.node_num, 1), \
            f"❌ Y shape wrong: {Y.shape}"
        assert TE.shape == (batch_size, self.input_len, 2), \
            f"❌ TE shape wrong: {TE.shape}"
        
        return {
            'X': torch.from_numpy(X),
            'Y': torch.from_numpy(Y),
            'TE': torch.from_numpy(TE)
        }
    
    def create_dataloader(self):
        """创建 DataLoader"""
        
        # 加载位置信息
        if self.use_location:
            self._load_location_dict()
        
        # 构建表路径
        table_path = f"odps://{self.odps_project}/tables/{self.odps_table}"
        
        if self.log:
            log_string(self.log, f'\nCreating dataset from: {table_path}')
            log_string(self.log, f'Slice: {self.slice_id}/{self.slice_count}')
        
        # 创建数据集
        dataset = OdpsTableDataset(
            table_path,
            slice_id=self.slice_id,
            slice_count=self.slice_count
        )
        
        if self.log:
            log_string(self.log, f'Dataset created with {len(dataset)} records')
        
        # 创建 DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        return data_loader
    
    def get_node_info(self):
        """获取节点信息"""
        return {
            'node_list': self.node_list,
            'node_num': self.node_num,
            'node_to_idx': self.node_to_idx,
            'node_locations': self.node_locations
        }


# ========== 便捷函数 ==========

def create_odps_table_dataloader(config, log=None):
    """
    便捷函数：创建 ODPS Table DataLoader
    
    使用示例:
        config = {
            'odps_project': 'autonavi_traffic_report',
            'odps_table': 'tb_inter_spatial_method_pretrain_data',
            'odps_meta_table': 'intersection_meta_1',
            'adcode': '650100',
            'batch_size': 64,
            'num_workers': 4,
            'input_len': 12
        }
        
        loader_wrapper = create_odps_table_dataloader(config, log)
        data_loader = loader_wrapper.create_dataloader()
        
        for batch in data_loader:
            X = batch['X']    # (batch, input_len, nodes, 1)
            Y = batch['Y']    # (batch, 1, nodes, 1)
            TE = batch['TE']  # (batch, input_len, 2)
    """
    return ODPSTableDataLoader(config, log)
