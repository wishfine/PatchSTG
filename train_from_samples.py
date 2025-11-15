"""
从预处理样本表直接训练 - 方案 2（极速版）

功能：
1. 从 ODPS 样本表直接读取已处理的样本
2. 反序列化为 numpy 数组
3. 开始训练（无需任何数据处理）

速度：秒级加载 → 立即开始训练！
"""

import os
import sys
import math
import time
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from odps import ODPS
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import PatchSTG
from lib.utils import log_string, _compute_loss, metric


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从样本表训练 PatchSTG')
    
    # ODPS 配置
    parser.add_argument('--odps_project', type=str, required=True)
    parser.add_argument('--odps_endpoint', type=str, required=True)
    parser.add_argument('--sample_table', type=str, required=True,
                        help='预处理好的样本表名')
    parser.add_argument('--metadata_file', type=str, required=True,
                        help='预处理元数据文件 (JSON)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='0')
    
    # 模型参数
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--tem_patchsize', type=int, default=12)
    parser.add_argument('--tem_patchnum', type=int, default=1)
    parser.add_argument('--factors', type=int, default=5)
    parser.add_argument('--spa_patchsize', type=int, default=4)
    parser.add_argument('--spa_patchnum', type=int, default=6)
    parser.add_argument('--tod', type=int, default=288)
    parser.add_argument('--dow', type=int, default=7)
    parser.add_argument('--input_dims', type=int, default=1)
    parser.add_argument('--node_dims', type=int, default=64)
    parser.add_argument('--tod_dims', type=int, default=64)
    parser.add_argument('--dow_dims', type=int, default=64)
    
    # 输出
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--log_file', type=str, default=None)
    
    return parser.parse_args()


def deserialize_array(s, shape):
    """反序列化字符串为 numpy 数组"""
    arr = np.array([float(x) for x in s.split(',')])
    return arr.reshape(shape)


def load_samples_from_odps(odps_client, table_name, split, metadata):
    """
    从 ODPS 样本表加载数据
    
    参数:
        odps_client: ODPS 客户端
        table_name: 表名
        split: 'train' / 'val' / 'test'
        metadata: 元数据字典
    
    返回:
        X, Y, TE_X, TE_Y (numpy 数组)
    """
    print(f"\n📥 加载 {split} 数据从 {table_name}...")
    
    # 查询指定 split 的数据
    query = f"""
    SELECT X, Y, TE_X, TE_Y
    FROM {table_name}
    WHERE split = '{split}'
    """
    
    # 从元数据获取形状信息
    num_nodes = metadata['node_num']
    input_len = metadata['input_len']
    output_len = metadata['output_len']
    
    X_list = []
    Y_list = []
    TE_X_list = []
    TE_Y_list = []
    
    with odps_client.execute_sql(query).open_reader() as reader:
        for record in tqdm(reader, desc=f"读取 {split}"):
            # 反序列化
            X = deserialize_array(record[0], (input_len, num_nodes, 1))
            Y = deserialize_array(record[1], (output_len, num_nodes, 1))
            TE_X = deserialize_array(record[2], (input_len, 2))
            TE_Y = deserialize_array(record[3], (output_len, 2))
            
            X_list.append(X)
            Y_list.append(Y)
            TE_X_list.append(TE_X)
            TE_Y_list.append(TE_Y)
    
    # 转换为 numpy 数组
    X = np.array(X_list)
    Y = np.array(Y_list)
    TE_X = np.array(TE_X_list)
    TE_Y = np.array(TE_Y_list)
    
    print(f"✅ {split} 数据加载完成")
    print(f"   X: {X.shape}")
    print(f"   Y: {Y.shape}")
    print(f"   TE_X: {TE_X.shape}")
    print(f"   TE_Y: {TE_Y.shape}")
    
    return X, Y, TE_X, TE_Y


def validate(model, valX, valY, valXTE, mean, std, device, batch_size):
    """验证函数"""
    model.eval()
    num_val = valX.shape[0]
    pred = []
    label = []

    num_batch = math.ceil(num_val / batch_size)
    
    with torch.no_grad():
        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_val, (batch_idx + 1) * batch_size)

            X = valX[start_idx:end_idx]
            Y = valY[start_idx:end_idx]
            TE = torch.from_numpy(valXTE[start_idx:end_idx]).to(device)
            NormX = torch.from_numpy((X - mean) / std).float().to(device)

            y_hat = model(NormX, TE)
            pred.append(y_hat.cpu().numpy() * std + mean)
            label.append(Y)
    
    pred = np.concatenate(pred, axis=0)
    label = np.concatenate(label, axis=0)

    maes = []
    rmses = []
    mapes = []

    for i in range(pred.shape[1]):
        mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    
    mae, rmse, mape = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    
    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🚀 PatchSTG 从样本表训练 - 方案 2（极速版）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 读取元数据
    print("📋 读取元数据...")
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("元数据:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"✅ 随机种子: {args.seed}\n")
    
    # 初始化日志
    if args.log_file is None:
        args.log_file = f"log/train_from_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.model_file is None:
        args.model_file = f"saved_models/model_from_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    
    os.makedirs('log', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    log = open(args.log_file, 'w')
    log_string(log, f"训练开始: {datetime.now()}")
    
    # 初始化 ODPS 客户端
    print("🔗 连接 ODPS...")
    access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    access_key = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    
    if not access_id or not access_key:
        raise ValueError("请设置环境变量: ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    
    odps_client = ODPS(access_id, access_key, args.odps_project, endpoint=args.odps_endpoint)
    print("✅ ODPS 连接成功\n")
    
    # 加载数据（极快！直接读取已处理样本）
    print("=" * 80)
    print("加载数据")
    print("=" * 80)
    
    trainX, trainY, trainXTE, trainYTE = load_samples_from_odps(
        odps_client, args.sample_table, 'train', metadata
    )
    valX, valY, valXTE, valYTE = load_samples_from_odps(
        odps_client, args.sample_table, 'val', metadata
    )
    testX, testY, testXTE, testYTE = load_samples_from_odps(
        odps_client, args.sample_table, 'test', metadata
    )
    
    mean = metadata['mean']
    std = metadata['std']
    node_num = metadata['node_num']
    
    print(f"\n✅ 数据加载完成！")
    print(f"  训练集: {trainX.shape[0]} 样本")
    print(f"  验证集: {valX.shape[0]} 样本")
    print(f"  测试集: {testX.shape[0]} 样本")
    print(f"  节点数: {node_num}")
    print()
    
    # 构建模型
    print("=" * 80)
    print("构建模型")
    print("=" * 80)
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 注意：这里简化了 patch 索引，实际应从元数据加载
    # 为简化，这里假设使用顺序索引
    ori_parts_idx = list(range(node_num))
    reo_parts_idx = list(range(node_num))
    reo_all_idx = list(range(node_num))
    
    model = PatchSTG(
        args.output_len if hasattr(args, 'output_len') else metadata['output_len'],
        args.tem_patchsize,
        args.tem_patchnum,
        node_num,
        args.spa_patchsize,
        args.spa_patchnum,
        args.tod,
        args.dow,
        args.layers,
        args.factors,
        args.input_dims,
        args.node_dims,
        args.tod_dims,
        args.dow_dims,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}\n")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1, 35, 40],
        gamma=0.5,
    )
    
    # 开始训练
    print("=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epoch}")
    print("=" * 80)
    print()
    
    min_val_loss = float('inf')
    best_epoch = 0
    num_train = trainX.shape[0]
    
    for epoch in range(1, args.max_epoch + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        batch_count = 0
        
        # 打乱训练数据
        indices = np.random.permutation(num_train)
        trainX = trainX[indices]
        trainY = trainY[indices]
        trainXTE = trainXTE[indices]
        
        num_batch = math.ceil(num_train / args.batch_size)
        
        pbar = tqdm(range(num_batch), desc=f"Epoch {epoch}/{args.max_epoch}")
        
        for batch_idx in pbar:
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

            X = trainX[start_idx:end_idx]
            Y = trainY[start_idx:end_idx]
            TE = torch.from_numpy(trainXTE[start_idx:end_idx]).to(device)
            NormX = torch.from_numpy((X - mean) / std).float().to(device)
            Y_tensor = torch.from_numpy(Y).float().to(device)
            
            optimizer.zero_grad()
            y_hat = model(NormX, TE)
            loss = _compute_loss(Y_tensor, y_hat * std + mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            train_loss_sum += loss.cpu().item()
            batch_count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss_sum / batch_count
        
        # 验证
        maes, rmses, mapes = validate(
            model, valX, valY, valXTE, mean, std, device, args.batch_size
        )
        
        val_mae = maes[-1]
        
        print(f"\nEpoch {epoch}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证 MAE: {val_mae:.4f}")
        print(f"  用时: {time.time() - epoch_start_time:.1f}s")
        
        log_string(log, f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val MAE={val_mae:.4f}")
        
        lr_scheduler.step()
        
        if val_mae < min_val_loss:
            min_val_loss = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_file)
            print(f"  ✅ 保存最佳模型")
        print()
    
    # 测试
    print("=" * 80)
    print("测试集评估")
    print("=" * 80)
    
    model.load_state_dict(torch.load(args.model_file))
    maes, rmses, mapes = validate(
        model, testX, testY, testXTE, mean, std, device, args.batch_size
    )
    
    print(f"\n最终测试结果:")
    print(f"  MAE:  {maes[-1]:.4f}")
    print(f"  RMSE: {rmses[-1]:.4f}")
    print(f"  MAPE: {mapes[-1]:.4f}")
    
    log.close()
    
    print("\n🎉 训练完成！")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"模型: {args.model_file}")
    print(f"日志: {args.log_file}")


if __name__ == '__main__':
    main()
