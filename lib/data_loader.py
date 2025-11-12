"""
数据加载与预处理模块
将数据加载逻辑从训练流程中分离，提供独立的数据处理接口
"""
import numpy as np
from lib.utils import loadData, log_string


class DataLoader:
    """
    数据加载器类，负责加载、预处理和管理训练/验证/测试数据
    """
    
    def __init__(self, config, log=None):
        """
        初始化数据加载器
        
        参数:
            config (dict): 包含数据相关配置的字典
            log: 日志文件对象（可选）
        """
        self.config = config
        self.log = log
        
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
        
        # Patch 索引
        self.ori_parts_idx = None
        self.reo_parts_idx = None
        self.reo_all_idx = None
        
        # 是否已加载数据
        self._loaded = False
    
    def load_data(self):
        """
        加载数据集
        """
        if self._loaded:
            if self.log:
                log_string(self.log, 'Data already loaded, skipping...')
            return
        
        if self.log:
            log_string(self.log, '\n------------ Loading Data -------------')
        
        # 调用原有的 loadData 函数
        (self.trainX, self.trainY, self.trainXTE, self.trainYTE,
         self.valX, self.valY, self.valXTE, self.valYTE,
         self.testX, self.testY, self.testXTE, self.testYTE,
         self.mean, self.std,
         self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx) = loadData(
            self.config['traffic_file'],
            self.config['meta_file'],
            self.config['input_len'],
            self.config['output_len'],
            self.config['train_ratio'],
            self.config['test_ratio'],
            self.config['adj_file'],
            self.config['recur_times'],
            self.config['tod'],
            self.config['dow'],
            self.config['spa_patchsize'],
            self.log
        )
        
        self._loaded = True
        
        if self.log:
            log_string(self.log, f'Train samples: {self.trainX.shape[0]}')
            log_string(self.log, f'Val samples: {self.valX.shape[0]}')
            log_string(self.log, f'Test samples: {self.testX.shape[0]}')
            log_string(self.log, f'Mean: {self.mean:.4f}, Std: {self.std:.4f}')
            log_string(self.log, '------------ End -------------\n')
    
    def get_train_data(self):
        """
        获取训练数据
        
        返回:
            tuple: (trainX, trainY, trainXTE, trainYTE)
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.trainX, self.trainY, self.trainXTE, self.trainYTE
    
    def get_val_data(self):
        """
        获取验证数据
        
        返回:
            tuple: (valX, valY, valXTE, valYTE)
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.valX, self.valY, self.valXTE, self.valYTE
    
    def get_test_data(self):
        """
        获取测试数据
        
        返回:
            tuple: (testX, testY, testXTE, testYTE)
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.testX, self.testY, self.testXTE, self.testYTE
    
    def get_normalization_params(self):
        """
        获取归一化参数
        
        返回:
            tuple: (mean, std)
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.mean, self.std
    
    def get_patch_indices(self):
        """
        获取 patch 索引
        
        返回:
            tuple: (ori_parts_idx, reo_parts_idx, reo_all_idx)
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx
    
    def shuffle_train_data(self, seed=None):
        """
        打乱训练数据
        
        参数:
            seed (int): 随机种子（可选）
        """
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
        """
        对数据进行归一化
        
        参数:
            data (np.ndarray): 待归一化的数据
            
        返回:
            np.ndarray: 归一化后的数据
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return (data - self.mean) / self.std
    
    def denormalize_data(self, data):
        """
        对数据进行反归一化
        
        参数:
            data (np.ndarray or torch.Tensor): 待反归一化的数据
            
        返回:
            归一化前的数据（类型与输入相同）
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return data * self.std + self.mean
    
    def get_data_info(self):
        """
        获取数据集的基本信息
        
        返回:
            dict: 包含数据集信息的字典
        """
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
            'num_nodes': self.trainX.shape[2]
        }
