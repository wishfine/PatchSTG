"""
从 MaxCompute 预处理样本表训练 PatchSTG 模型

与 main.py 的区别:
1. 数据来源: tb_patchstg_train_samples_full (预处理表) 而非 npz 文件
2. 时间特征: 6维 (week, hour, minute, day_type, day, month) 而非 2维 (tod, dow)
3. 城市范围: 混合多城市训练
4. 数据格式: 已经是滑动窗口样本,不需要 seq2instance

使用方法:
    python train_from_samples.py --config config/samples.conf --cuda 0

作者: AI Assistant
日期: 2025-11-15
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm
import os

# 引入模型与工具函数
from models.model import PatchSTG
from lib.utils import log_string, _compute_loss, metric, loadDataFromSamples
from lib.patchstg_sample_loader import PatchSTGSampleLoader, build_locations_from_pairs


class SampleSolver(object):
    """
    使用预处理样本训练 PatchSTG 的 Solver
    
    主要改动:
    - 使用 PatchSTGSampleLoader 加载数据
    - 支持 6 维时间特征
    - 模型需要适配新的时间特征维度
    """
    
    def __init__(self, config):
        """
        参数:
            config (dict): 配置参数字典
        """
        self.__dict__.update(**config)
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self._build_timestamped_log_path(self.log_file, self.run_timestamp)
        
        # 初始化日志
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        global log
        log = open(self.log_file, 'w')
        log_string(log, "======================CONFIG======================")
        for key, value in config.items():
            log_string(log, f"{key}: {value}")
        log_string(log, "==================================================")
        log_string(log, f"日志文件: {self.log_file}")
        
        # 加载数据
        self.load_sample_data()
        
        # 设备选择
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        log_string(log, f"使用设备: {self.device}")
        
        # 构建模型
        self.build_model()
        
        self.best_epoch = 0

    def _build_timestamped_log_path(self, base_path, timestamp):
        """根据启动时间生成带时间戳的日志文件路径"""
        if not base_path:
            base_path = os.path.join('log', 'patchstg.log')
        log_dir = os.path.dirname(base_path) or '.'
        filename = os.path.basename(base_path) or 'patchstg.log'
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = '.log'
        timestamped_name = f"{name}_{timestamp}{ext}"
        return os.path.join(log_dir, timestamped_name)
    
    def load_sample_data(self):
        """
        使用 PatchSTGSampleLoader 加载样本数据
        """
        log_string(log, "======================LOADING DATA======================")
        
        # 初始化数据加载器
        loader = PatchSTGSampleLoader(
            odps_project=self.odps_project,
            odps_table=self.odps_table,
            ds_partition=self.ds_partition,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            use_6dim_time_feat=getattr(self, 'use_6dim_time_feat', False),
        )
        
        # 加载样本数据
        if hasattr(self, 'debug_csv') and self.debug_csv:
            # 调试模式: 从本地CSV加载
            log_string(log, f"调试模式: 从CSV加载数据: {self.debug_csv}")
            loader.load_from_csv(self.debug_csv)
        else:
            # 生产模式: 从MaxCompute加载
            log_string(log, f"从 MaxCompute 加载数据: {self.odps_table}, ds={self.ds_partition}")
            loader.load_from_odps(limit=getattr(self, 'data_limit', None))
        
        # 加载路口位置信息
        if hasattr(self, 'inter_location_dict') and self.inter_location_dict:
            # 从字典加载 (用于调试)
            loader.load_inter_locations(self.inter_location_dict)
        elif hasattr(self, 'inter_location_table'):
            # 从 MaxCompute 表加载
            loader.load_inter_locations_from_odps(self.inter_location_table)
        else:
            log_string(log, "警告: 未提供路口位置信息,将无法进行空间划分")
        
        # 准备训练数据
        data_dict = loader.prepare_training_data(normalize=False)
        self.sample_metadata = data_dict['metadata']
        
        # 构建 locations 矩阵 (用于KDTree)
        unique_pairs = self.sample_metadata['node_pairs']
        if loader.inter_id_locations:
            locations = build_locations_from_pairs(unique_pairs, loader.inter_id_locations)
            log_string(log, f"构建 locations 矩阵: shape={locations.shape}")
        else:
            # 如果没有位置信息,使用随机位置 (仅用于调试)
            log_string(log, "警告: 使用随机位置进行空间划分 (仅调试)")
            num_pairs = len(unique_pairs)
            locations = np.random.rand(2, num_pairs).astype(np.float32)
        
        # 使用 loadDataFromSamples 进行空间划分
        adjpath = os.path.join(self.model_file.replace('.pth', '_adj.npy'))
        result = loadDataFromSamples(
            data_dict,
            locations,
            adjpath,
            self.recurtimes,
            self.spa_patchsize,
            log
        )
        
        # 解包结果
        (self.trainX, self.trainY, self.trainXTE, self.trainYTE,
         self.valX, self.valY, self.valXTE, self.valYTE,
         self.testX, self.testY, self.testXTE, self.testYTE,
         self.mean, self.std,
         self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx) = result
        
        # 更新节点数量 (实际的最大节点数)
        self.node_num = self.trainX.shape[2]
        log_string(log, f"节点数量 (max_nodes): {self.node_num}")
        
        # 时间特征维度
        self.time_feat_dim = self.sample_metadata.get('time_feat_dim', 2)
        log_string(log, f"时间特征维度: {self.time_feat_dim}")
        log_string(log, "=======================================================")
    
    def build_model(self):
        """
        构建模型、优化器与学习率调度器
        
        注意: 需要修改模型以支持6维时间特征
        """
        log_string(log, "======================BUILD MODEL======================")
        
        # 实例化模型
        # 注意: 原始 PatchSTG 假设时间特征是 2 维 (tod, dow)
        # 现在需要传入 6 维特征,可能需要修改模型代码
        # 这里先使用原始模型,后续可能需要调整
        
        try:
            self.model = PatchSTG(
                self.output_len,
                self.tem_patchsize,
                self.tem_patchnum,
                self.node_num,
                self.spa_patchsize,
                self.spa_patchnum,
                # 时间特征相关 - 保持兼容性
                self.tod if hasattr(self, 'tod') else 24,  # tod 默认 24
                self.dow if hasattr(self, 'dow') else 7,   # dow 默认 7
                self.layers,
                self.factors,
                self.input_dims,
                self.node_dims,
                # 时间嵌入维度 - 需要根据6维调整
                getattr(self, 'tod_dims', 8),  # 可以复用或调整
                getattr(self, 'dow_dims', 8),
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx
            ).to(self.device)
            
            log_string(log, f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        except Exception as e:
            log_string(log, f"模型构建失败: {e}")
            log_string(log, "可能需要修改 PatchSTG 模型以支持 6 维时间特征")
            raise
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=getattr(self, 'milestones', [1, 35, 40]),
            gamma=getattr(self, 'lr_gamma', 0.5),
        )
        
        log_string(log, "=======================================================")
    
    def vali(self):
        """验证模式"""
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []
        
        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
                
                X = self.valX[start_idx : end_idx]
                Y = self.valY[start_idx : end_idx]
                TE = torch.from_numpy(self.valXTE[start_idx : end_idx]).to(self.device)
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                
                y_hat = self.model(NormX, TE)
                
                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(Y)
        
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        
        # 计算指标
        maes, rmses, mapes = [], [], []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, f'step {i+1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, f'average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
    
    def train(self):
        """训练模式"""
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]
        
        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()
            
            # 随机打乱训练数据
            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainXTE = self.trainXTE[permutation]
            self.trainYTE = self.trainYTE[permutation]
            
            num_batch = math.ceil(num_train / self.batch_size)
            
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_train, (batch_idx + 1) * self.batch_size)
                
                X = self.trainX[start_idx : end_idx]
                Y = self.trainY[start_idx : end_idx]
                TE = torch.from_numpy(self.trainXTE[start_idx : end_idx]).to(self.device)
                
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                NormY = torch.from_numpy((Y - self.mean) / self.std).float().to(self.device)
                
                self.optimizer.zero_grad()
                y_hat = self.model(NormX, TE)
                loss = _compute_loss(NormY, y_hat)
                
                loss.backward()
                self.optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
            
            self.lr_scheduler.step()
            
            # 验证
            log_string(log, f'epoch {epoch}, lr {self.optimizer.param_groups[0]["lr"]:.6f}, '
                           f'train loss {train_l_sum / batch_count:.6f}, time {time.time() - start:.1f}s')
            
            maes, rmses, mapes = self.vali()
            
            # 保存最优模型
            if maes[-1] < min_loss:
                min_loss = maes[-1]
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_file)
                log_string(log, f'保存模型到: {self.model_file}')
        
        log_string(log, f'最佳 epoch: {self.best_epoch}')
        log_string(log, "======================TRAIN END======================")
    
    def test(self):
        """测试模式"""
        log_string(log, "======================TEST MODE======================")
        
        # 加载最优模型
        self.model.load_state_dict(torch.load(self.model_file))
        log_string(log, f'加载模型: {self.model_file}')
        
        self.model.eval()
        num_test = self.testX.shape[0]
        pred = []
        label = []
        
        num_batch = math.ceil(num_test / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_test, (batch_idx + 1) * self.batch_size)
                
                X = self.testX[start_idx : end_idx]
                Y = self.testY[start_idx : end_idx]
                TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                
                y_hat = self.model(NormX, TE)
                
                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(Y)
        
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        
        # 计算指标
        maes, rmses, mapes = [], [], []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, f'step {i+1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        mae, rmse, mape = metric(pred, label)
        log_string(log, f'average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        log_string(log, "=======================================================")


def parse_config(config_file):
    """解析配置文件"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 提取配置参数
    params = {}
    
    # Data
    params['odps_project'] = config.get('Data', 'odps_project')
    params['odps_table'] = config.get('Data', 'odps_table')
    params['ds_partition'] = config.get('Data', 'ds_partition')
    params['train_ratio'] = config.getfloat('Data', 'train_ratio', fallback=0.7)
    params['val_ratio'] = config.getfloat('Data', 'val_ratio', fallback=0.1)
    params['use_6dim_time_feat'] = config.getboolean('Data', 'use_6dim_time_feat', fallback=False)
    
    if config.has_option('Data', 'inter_location_table'):
        params['inter_location_table'] = config.get('Data', 'inter_location_table')
    
    if config.has_option('Data', 'debug_csv'):
        params['debug_csv'] = config.get('Data', 'debug_csv')
    
    # Model
    params['input_len'] = config.getint('Model', 'input_len', fallback=12)
    params['output_len'] = config.getint('Model', 'output_len', fallback=12)
    params['tem_patchsize'] = config.getint('Model', 'tem_patchsize')
    params['tem_patchnum'] = config.getint('Model', 'tem_patchnum')
    params['spa_patchsize'] = config.getint('Model', 'spa_patchsize')
    params['spa_patchnum'] = config.getint('Model', 'spa_patchnum')
    params['layers'] = config.getint('Model', 'layers')
    params['factors'] = config.getint('Model', 'factors')
    params['input_dims'] = config.getint('Model', 'input_dims')
    params['node_dims'] = config.getint('Model', 'node_dims')
    params['recurtimes'] = config.getint('Model', 'recurtimes')
    
    # Training
    params['learning_rate'] = config.getfloat('Training', 'learning_rate')
    params['weight_decay'] = config.getfloat('Training', 'weight_decay')
    params['batch_size'] = config.getint('Training', 'batch_size')
    params['max_epoch'] = config.getint('Training', 'max_epoch')
    
    # Log
    params['log_file'] = config.get('Log', 'log_file')
    params['model_file'] = config.get('Log', 'model_file')
    
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--cuda', type=int, default=0, help='GPU 设备编号')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    
    args = parser.parse_args()
    
    # 解析配置
    config = parse_config(args.config)
    config['cuda'] = args.cuda
    
    # 创建 Solver
    solver = SampleSolver(config)
    
    # 运行
    if args.mode == 'train':
        solver.train()
        solver.test()
    else:
        solver.test()
    
    log.close()
