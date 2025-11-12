#!/usr/bin/env python
"""
ODPS 数据检查脚本
在训练前检查数据质量和统计信息
"""
import argparse
import configparser
import json
from lib.odps_data_loader import ODPSDataLoader


def check_odps_data(config_file, output_file='odps_data_info.json', sample_size=5):
    """
    检查 ODPS 数据并生成报告
    
    参数:
        config_file: 配置文件路径
        output_file: 输出的数据信息文件
        sample_size: 显示的样本数量
    """
    # 读取配置
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 构建配置字典
    data_config = {
        'odps_project': config['data']['odps_project'],
        'odps_endpoint': config['data']['odps_endpoint'],
        'odps_table': config['data']['odps_table'],
        'adcode': config['data']['adcode'],
        'start_date': config['data']['start_date'],
        'end_date': config['data']['end_date'],
        'input_len': int(config['data']['input_len']),
        'output_len': int(config['data']['output_len']),
        'train_ratio': float(config['data']['train_ratio']),
        'val_ratio': float(config['data']['val_ratio']),
        'test_ratio': float(config['data']['test_ratio']),
    }
    
    # 创建日志
    log_file = config['file']['log'] + '_check'
    log = open(log_file, 'w')
    
    try:
        print("=" * 80)
        print("ODPS 数据检查")
        print("=" * 80)
        print()
        
        print(f"配置文件: {config_file}")
        print(f"城市代码: {data_config['adcode']}")
        print(f"日期范围: {data_config['start_date']} ~ {data_config['end_date']}")
        print()
        
        # 创建数据加载器
        print("正在连接 ODPS 并加载数据...")
        data_loader = ODPSDataLoader(data_config, log)
        data_loader.load_data()
        
        # 获取数据信息
        info = data_loader.get_data_info()
        
        # 打印基本信息
        print("\n" + "=" * 80)
        print("数据统计")
        print("=" * 80)
        print()
        
        print(f"节点数量: {info['num_nodes']}")
        print(f"训练样本: {info['train_samples']}")
        print(f"验证样本: {info['val_samples']}")
        print(f"测试样本: {info['test_samples']}")
        print(f"总样本数: {info['train_samples'] + info['val_samples'] + info['test_samples']}")
        print()
        
        print(f"输入形状: {info['input_shape']}")
        print(f"输出形状: {info['output_shape']}")
        print()
        
        print(f"归一化参数:")
        print(f"  Mean: {info['mean']:.4f}")
        print(f"  Std:  {info['std']:.4f}")
        print()
        
        # 节点样本
        if 'node_list' in info and info['node_list']:
            print(f"节点样本 (前 {min(sample_size, len(info['node_list']))} 个):")
            for i, (nds_id, next_nds_id) in enumerate(info['node_list'][:sample_size]):
                print(f"  {i+1}. {nds_id} -> {next_nds_id}")
            if len(info['node_list']) > sample_size:
                print(f"  ... 还有 {len(info['node_list']) - sample_size} 个节点")
        print()
        
        # 数据形状分析
        print("=" * 80)
        print("数据形状分析")
        print("=" * 80)
        print()
        
        trainX, trainY, trainXTE, trainYTE = data_loader.get_train_data()
        print(f"trainX:   {trainX.shape}  - (样本数, 输入长度, 节点数, 特征维度)")
        print(f"trainY:   {trainY.shape}  - (样本数, 输出长度, 节点数, 特征维度)")
        print(f"trainXTE: {trainXTE.shape}  - (样本数, 输入长度, 时间特征维度)")
        print(f"trainYTE: {trainYTE.shape}  - (样本数, 输出长度, 时间特征维度)")
        print()
        
        # 数据范围检查
        print("=" * 80)
        print("数据范围检查")
        print("=" * 80)
        print()
        
        import numpy as np
        
        print(f"训练集流量统计:")
        print(f"  最小值: {np.min(trainX):.4f}")
        print(f"  最大值: {np.max(trainX):.4f}")
        print(f"  均值:   {np.mean(trainX):.4f}")
        print(f"  中位数: {np.median(trainX):.4f}")
        print(f"  标准差: {np.std(trainX):.4f}")
        print()
        
        # 零值统计
        zero_count = np.sum(trainX == 0)
        total_count = trainX.size
        zero_ratio = zero_count / total_count * 100
        print(f"零值统计:")
        print(f"  零值数量: {zero_count}")
        print(f"  总数量:   {total_count}")
        print(f"  零值比例: {zero_ratio:.2f}%")
        print()
        
        # 保存信息到 JSON
        info['statistics'] = {
            'min': float(np.min(trainX)),
            'max': float(np.max(trainX)),
            'mean': float(np.mean(trainX)),
            'median': float(np.median(trainX)),
            'std': float(np.std(trainX)),
            'zero_count': int(zero_count),
            'total_count': int(total_count),
            'zero_ratio': float(zero_ratio),
        }
        
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print("=" * 80)
        print("检查完成")
        print("=" * 80)
        print()
        print(f"✓ 日志文件: {log_file}")
        print(f"✓ 数据信息: {output_file}")
        print()
        print("建议:")
        
        # 给出建议
        if info['num_nodes'] < 100:
            print("  ⚠️  节点数较少，考虑扩大日期范围或区域")
        
        if info['train_samples'] < 1000:
            print("  ⚠️  训练样本较少，建议增加数据量")
        
        if zero_ratio > 50:
            print("  ⚠️  零值比例过高，检查数据质量")
        
        if info['std'] < 1:
            print("  ⚠️  标准差较小，数据可能缺乏变化")
        
        print("\n准备就绪！可以开始训练:")
        print(f"  python train_odps.py --config {config_file} --mode train")
        print()
        
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查 ODPS 数据')
    parser.add_argument('--config', type=str, default='config/ODPS.conf',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='odps_data_info.json',
                        help='输出的数据信息文件')
    parser.add_argument('--sample-size', type=int, default=5,
                        help='显示的节点样本数量')
    
    args = parser.parse_args()
    
    check_odps_data(args.config, args.output, args.sample_size)
