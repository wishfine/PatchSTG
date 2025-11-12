"""
使用预加载数据进行训练的示例
展示如何分离数据加载和训练流程
"""
import argparse
import configparser
import random
import torch
import numpy as np
from lib.data_loader import DataLoader
from lib.utils import log_string
from main import Solver


def train_with_preloaded_data(config_file):
    """
    使用预加载的数据进行训练
    
    参数:
        config_file (str): 配置文件路径
    """
    # 读取配置
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 解析配置参数
    args_dict = {
        # 训练参数
        'cuda': config['train']['cuda'],
        'seed': int(config['train']['seed']),
        'batch_size': int(config['train']['batch_size']),
        'max_epoch': int(config['train']['max_epoch']),
        'learning_rate': float(config['train']['learning_rate']),
        'weight_decay': float(config['train']['weight_decay']),
        
        # 数据参数
        'input_len': int(config['data']['input_len']),
        'output_len': int(config['data']['output_len']),
        'train_ratio': float(config['data']['train_ratio']),
        'val_ratio': float(config['data']['val_ratio']),
        'test_ratio': float(config['data']['test_ratio']),
        
        # 模型参数
        'layers': int(config['param']['layers']),
        'tem_patchsize': int(config['param']['tps']),
        'tem_patchnum': int(config['param']['tpn']),
        'factors': int(config['param']['factors']),
        'recur_times': int(config['param']['recur']),
        'spa_patchsize': int(config['param']['sps']),
        'spa_patchnum': int(config['param']['spn']),
        'node_num': int(config['param']['nodes']),
        'tod': int(config['param']['tod']),
        'dow': int(config['param']['dow']),
        'input_dims': int(config['param']['id']),
        'node_dims': int(config['param']['nd']),
        'tod_dims': int(config['param']['td']),
        'dow_dims': int(config['param']['dd']),
        
        # 文件路径
        'traffic_file': config['file']['traffic'],
        'meta_file': config['file']['meta'],
        'adj_file': config['file']['adj'],
        'model_file': config['file']['model'],
        'log_file': config['file']['log'],
    }
    
    # 打开日志文件
    log = open(args_dict['log_file'], 'w')
    
    try:
        # 设置随机种子
        if args_dict['seed'] is not None:
            random.seed(args_dict['seed'])
            np.random.seed(args_dict['seed'])
            torch.manual_seed(args_dict['seed'])
            torch.cuda.manual_seed(args_dict['seed'])
            torch.backends.cudnn.deterministic = True
        
        # 记录配置
        log_string(log, '------------ Options -------------')
        for k, v in args_dict.items():
            log_string(log, f'{k}: {v}')
        log_string(log, '-------------- End ----------------')
        
        # 步骤 1: 预先加载数据（可以在训练之前完成）
        log_string(log, '\n========== Step 1: Loading Data ==========')
        data_loader = DataLoader(args_dict, log)
        data_loader.load_data()
        
        # 打印数据信息
        data_info = data_loader.get_data_info()
        log_string(log, '\nData loaded successfully!')
        log_string(log, f'Train samples: {data_info["train_samples"]}')
        log_string(log, f'Val samples: {data_info["val_samples"]}')
        log_string(log, f'Test samples: {data_info["test_samples"]}')
        
        # 步骤 2: 使用预加载的数据创建 Solver 并训练
        log_string(log, '\n========== Step 2: Training Model ==========')
        solver = Solver(args_dict, data_loader=data_loader)
        
        # 开始训练
        solver.train()
        
        # 步骤 3: 测试
        log_string(log, '\n========== Step 3: Testing Model ==========')
        solver.test()
        
        log_string(log, '\n========== Training Completed ==========')
        
    except Exception as e:
        log_string(log, f'\nError: {str(e)}')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用预加载数据进行训练')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (例如: config/CA.conf)')
    
    args = parser.parse_args()
    
    train_with_preloaded_data(args.config)
