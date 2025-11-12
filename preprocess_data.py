"""
独立的数据预处理脚本
用于在训练之前预加载和检查数据
"""
import argparse
import configparser
import json
from lib.data_loader import DataLoader
from lib.utils import log_string


def preprocess_and_save_info(config_file, output_file='data_info.json'):
    """
    预处理数据并保存数据集信息
    
    参数:
        config_file (str): 配置文件路径
        output_file (str): 输出的数据信息文件路径
    """
    # 读取配置
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 构建数据配置字典
    data_config = {
        'traffic_file': config['file']['traffic'],
        'meta_file': config['file']['meta'],
        'adj_file': config['file']['adj'],
        'input_len': int(config['data']['input_len']),
        'output_len': int(config['data']['output_len']),
        'train_ratio': float(config['data']['train_ratio']),
        'test_ratio': float(config['data']['test_ratio']),
        'recur_times': int(config['param']['recur']),
        'tod': int(config['param']['tod']),
        'dow': int(config['param']['dow']),
        'spa_patchsize': int(config['param']['sps']),
    }
    
    # 创建日志
    log_file = config['file']['log'].replace('.log', '_preprocess.log')
    log = open(log_file, 'w')
    
    try:
        log_string(log, '========== Data Preprocessing ==========')
        log_string(log, f'Config file: {config_file}')
        log_string(log, f'Output file: {output_file}')
        log_string(log, '')
        
        # 创建数据加载器并加载数据
        data_loader = DataLoader(data_config, log)
        data_loader.load_data()
        
        # 获取数据信息
        data_info = data_loader.get_data_info()
        
        # 打印数据信息
        log_string(log, '\n========== Data Information ==========')
        for key, value in data_info.items():
            log_string(log, f'{key}: {value}')
        log_string(log, '=' * 40)
        
        # 保存数据信息到 JSON 文件
        with open(output_file, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        log_string(log, f'\nData information saved to: {output_file}')
        log_string(log, 'Preprocessing completed successfully!')
        
        print(f'✓ 数据预处理完成！')
        print(f'  - 日志文件: {log_file}')
        print(f'  - 数据信息: {output_file}')
        print(f'\n数据集概览:')
        print(f'  训练样本: {data_info["train_samples"]}')
        print(f'  验证样本: {data_info["val_samples"]}')
        print(f'  测试样本: {data_info["test_samples"]}')
        print(f'  节点数量: {data_info["num_nodes"]}')
        print(f'  归一化参数 - Mean: {data_info["mean"]:.4f}, Std: {data_info["std"]:.4f}')
        
    except Exception as e:
        log_string(log, f'\nError during preprocessing: {str(e)}')
        print(f'✗ 预处理失败: {str(e)}')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--config', type=str, required=True, 
                        help='配置文件路径 (例如: config/CA.conf)')
    parser.add_argument('--output', type=str, default='data_info.json',
                        help='输出的数据信息文件路径 (默认: data_info.json)')
    
    args = parser.parse_args()
    
    preprocess_and_save_info(args.config, args.output)
