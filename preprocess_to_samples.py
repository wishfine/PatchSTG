"""
PatchSTG 数据预处理脚本 - 方案 2（终极版）

功能：
1. 从 ODPS 原始表读取流量数据（流式）
2. 转换为时间序列格式
3. 创建空间 patches（KD-tree 分组）
4. 生成训练/验证/测试样本
5. 保存到 ODPS 样本表（供训练直接读取）

运行一次，训练无限次！
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.odps_data_loader import ODPSDataLoader
from lib.utils import log_string


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预处理 PatchSTG 训练样本')
    
    # ODPS 配置
    parser.add_argument('--odps_project', type=str, required=True,
                        help='ODPS 项目名')
    parser.add_argument('--odps_endpoint', type=str, required=True,
                        help='ODPS endpoint')
    parser.add_argument('--odps_table', type=str, required=True,
                        help='原始数据表名')
    parser.add_argument('--odps_meta_table', type=str, required=True,
                        help='节点元数据表名')
    parser.add_argument('--output_table', type=str, required=True,
                        help='输出样本表名（不存在会自动创建）')
    
    # 数据过滤
    parser.add_argument('--adcode', type=str, default='110000',
                        help='城市代码（默认：110000 北京）')
    parser.add_argument('--start_date', type=str, required=True,
                        help='开始日期 YYYYMMDD')
    parser.add_argument('--end_date', type=str, required=True,
                        help='结束日期 YYYYMMDD')
    
    # 模型参数
    parser.add_argument('--input_len', type=int, default=12,
                        help='输入序列长度（默认：12）')
    parser.add_argument('--output_len', type=int, default=12,
                        help='输出序列长度（默认：12）')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='训练集比例（默认：0.6）')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例（默认：0.2）')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='测试集比例（默认：0.2）')
    
    # 空间参数
    parser.add_argument('--recur_times', type=int, default=1,
                        help='KD-tree 递归次数（默认：1）')
    parser.add_argument('--spa_patchsize', type=int, default=4,
                        help='空间 patch 大小（默认：4）')
    
    # 其他
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='写入 ODPS 的批次大小（默认：10000）')
    
    return parser.parse_args()


def save_samples_to_odps(odps_client, table_name, samples_df, batch_size=10000):
    """
    将样本保存到 ODPS 表
    
    参数:
        odps_client: ODPS 客户端
        table_name: 表名
        samples_df: 样本 DataFrame
        batch_size: 批次大小
    """
    from odps import TableSchema
    from odps.models import Column
    
    print(f"\n📝 准备写入 ODPS 表: {table_name}")
    
    # 定义表结构
    schema = TableSchema([
        Column('sample_id', 'string'),      # 样本ID
        Column('split', 'string'),          # train/val/test
        Column('X', 'string'),              # 输入序列 (序列化)
        Column('Y', 'string'),              # 输出序列 (序列化)
        Column('TE_X', 'string'),           # 输入时间特征 (序列化)
        Column('TE_Y', 'string'),           # 输出时间特征 (序列化)
        Column('node_indices', 'string'),   # 节点索引列表
        Column('timestamp', 'string'),      # 样本时间戳
    ])
    
    # 如果表不存在，创建
    if not odps_client.exist_table(table_name):
        print(f"   创建新表: {table_name}")
        odps_client.create_table(table_name, schema)
    
    table = odps_client.get_table(table_name)
    
    # 分批写入
    total_samples = len(samples_df)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"   总样本数: {total_samples}")
    print(f"   批次大小: {batch_size}")
    print(f"   总批次数: {num_batches}")
    print(f"\n开始写入...")
    
    with table.open_writer() as writer:
        for batch_idx in tqdm(range(num_batches), desc="写入进度"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_df = samples_df.iloc[start_idx:end_idx]
            
            # 转换为记录列表
            records = []
            for _, row in batch_df.iterrows():
                records.append([
                    row['sample_id'],
                    row['split'],
                    row['X'],
                    row['Y'],
                    row['TE_X'],
                    row['TE_Y'],
                    row['node_indices'],
                    row['timestamp'],
                ])
            
            writer.write(records)
    
    print(f"\n✅ 成功写入 {total_samples} 个样本到 {table_name}")


def serialize_array(arr):
    """将 numpy 数组序列化为字符串"""
    return ','.join(arr.flatten().astype(str))


def create_samples_dataframe(trainX, trainY, trainXTE, trainYTE, split='train'):
    """
    将 numpy 数组转换为 DataFrame
    
    参数:
        trainX: (N, T_in, num_nodes, C)
        trainY: (N, T_out, num_nodes, C)
        trainXTE: (N, T_in, 2)
        trainYTE: (N, T_out, 2)
        split: 'train' / 'val' / 'test'
    
    返回:
        DataFrame with columns: sample_id, split, X, Y, TE_X, TE_Y, node_indices, timestamp
    """
    num_samples = trainX.shape[0]
    num_nodes = trainX.shape[2]
    
    print(f"\n🔄 转换 {split} 数据为 DataFrame...")
    print(f"   样本数: {num_samples}")
    print(f"   节点数: {num_nodes}")
    
    samples = []
    for i in tqdm(range(num_samples), desc=f"处理 {split}"):
        sample_id = f"{split}_{i}"
        
        # 序列化数组
        X_str = serialize_array(trainX[i])        # (T_in, num_nodes, C)
        Y_str = serialize_array(trainY[i])        # (T_out, num_nodes, C)
        TE_X_str = serialize_array(trainXTE[i])   # (T_in, 2)
        TE_Y_str = serialize_array(trainYTE[i])   # (T_out, 2)
        
        # 节点索引
        node_indices_str = ','.join([str(j) for j in range(num_nodes)])
        
        # 时间戳（从时间特征提取）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        samples.append({
            'sample_id': sample_id,
            'split': split,
            'X': X_str,
            'Y': Y_str,
            'TE_X': TE_X_str,
            'TE_Y': TE_Y_str,
            'node_indices': node_indices_str,
            'timestamp': timestamp,
        })
    
    return pd.DataFrame(samples)


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🚀 PatchSTG 数据预处理 - 方案 2")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 配置
    config = {
        'odps_project': args.odps_project,
        'odps_endpoint': args.odps_endpoint,
        'odps_table': args.odps_table,
        'odps_meta_table': args.odps_meta_table,
        'adcode': args.adcode,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'input_len': args.input_len,
        'output_len': args.output_len,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'recur_times': args.recur_times,
        'spa_patchsize': args.spa_patchsize,
    }
    
    print("📋 配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 步骤 1: 使用 ODPSDataLoader 加载和处理数据
    print("=" * 80)
    print("步骤 1: 加载和处理原始数据")
    print("=" * 80)
    
    log_file = open(f'preprocess_{args.adcode}_{args.start_date}_{args.end_date}.log', 'w')
    data_loader = ODPSDataLoader(config, log_file)
    
    # 加载数据（会自动进行流式处理、时间序列转换、空间分组、样本生成）
    data_loader.load_data()
    
    # 获取处理后的数据
    trainX, trainY, trainXTE, trainYTE = data_loader.get_train_data()
    valX, valY, valXTE, valYTE = data_loader.get_val_data()
    testX, testY, testXTE, testYTE = data_loader.get_test_data()
    
    print("\n✅ 数据加载完成")
    print(f"  训练集: {trainX.shape[0]} 样本")
    print(f"  验证集: {valX.shape[0]} 样本")
    print(f"  测试集: {testX.shape[0]} 样本")
    
    # 步骤 2: 转换为 DataFrame
    print("\n" + "=" * 80)
    print("步骤 2: 转换为样本格式")
    print("=" * 80)
    
    train_df = create_samples_dataframe(trainX, trainY, trainXTE, trainYTE, 'train')
    val_df = create_samples_dataframe(valX, valY, valXTE, valYTE, 'val')
    test_df = create_samples_dataframe(testX, testY, testXTE, testYTE, 'test')
    
    # 合并所有数据
    all_samples_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"\n✅ 样本转换完成")
    print(f"  总样本数: {len(all_samples_df)}")
    print(f"  训练: {len(train_df)} ({len(train_df)/len(all_samples_df)*100:.1f}%)")
    print(f"  验证: {len(val_df)} ({len(val_df)/len(all_samples_df)*100:.1f}%)")
    print(f"  测试: {len(test_df)} ({len(test_df)/len(all_samples_df)*100:.1f}%)")
    
    # 步骤 3: 保存到 ODPS
    print("\n" + "=" * 80)
    print("步骤 3: 保存到 ODPS 样本表")
    print("=" * 80)
    
    odps_client = data_loader._odps_client
    save_samples_to_odps(
        odps_client,
        args.output_table,
        all_samples_df,
        batch_size=args.batch_size
    )
    
    # 保存元数据
    metadata = {
        'source_table': args.odps_table,
        'output_table': args.output_table,
        'adcode': args.adcode,
        'date_range': f"{args.start_date}~{args.end_date}",
        'total_samples': len(all_samples_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'node_num': trainX.shape[2],
        'input_len': args.input_len,
        'output_len': args.output_len,
        'mean': float(data_loader.mean),
        'std': float(data_loader.std),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    print("\n📊 预处理完成！元数据:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    
    # 保存元数据到本地
    import json
    metadata_file = f"metadata_{args.output_table}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n💾 元数据已保存: {metadata_file}")
    
    log_file.close()
    
    print("\n" + "=" * 80)
    print("🎉 预处理完成！")
    print("=" * 80)
    print(f"输出表: {args.output_table}")
    print(f"总样本: {len(all_samples_df)}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n下一步：使用 train_from_samples.py 进行训练")
    print("=" * 80)


if __name__ == '__main__':
    main()
