import os
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 工具模块：包含日志、评估指标、数据变换与分区/重排等函数

# 写日志并同时打印到 stdout，确保日志文件实时刷新，并带上时间戳
def log_string(log, string):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {string}"
    log.write(formatted + '\n')
    log.flush()
    print(formatted)


# 计算评估指标（针对 numpy 数组）
def metric(pred, label):
    """
    计算 MAE, RMSE, MAPE。对 label==0 的位置做掩码，避免除零或无意义统计。
    返回值为 (mae, rmse, mape) 三元组。
    """
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        # mask 标记 label 不为 0 的位置
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        # 归一化 mask，保证不同样本之间的可比较性
        mask /= np.mean(mask)

        # 绝对误差
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)

        # WAPE（加权绝对百分比误差），用于衡量总体误差占真实值的比例
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)

        # RMSE 需要平方后求均值再开根号
        rmse = np.square(mae)

        # MAPE（相对误差），在 label 为 0 时会产生 inf，需要 later 用 nan_to_num 处理
        mape = np.divide(mae, label)

        # 对于存在 NaN 的位置用 0 替代，然后按 mask 计算平均
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)

        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))

        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape


def masked_mae(preds, labels, null_val=np.nan):
    """
    用于训练时的损失计算：对指定的 null_val 或 NaN 位置进行掩码，然后计算平均绝对误差。
    - preds, labels: torch.Tensor
    - null_val: 如果为 np.nan，则按 labels 中的 NaN 判定掩码；否则按等于 null_val 判定
    """
    if np.isnan(null_val):
        # mask 为 labels 中非 NaN 的位置
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    # 归一化 mask，使得不同样本/不同批次之间的损失可比较
    mask /=  torch.mean((mask))
    # 将可能产生的 NaN 替换为 0（安全性处理）
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # 计算绝对误差并应用 mask
    loss = torch.abs(preds-labels)
    loss = loss * mask
    # 再次把 NaN 替换为 0，防止后续求均值出错
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def _compute_loss(y_true, y_predicted):
        # 在代码中统一把 (y_true, y_pred) 的位置传入 masked_mae
        return masked_mae(y_predicted, y_true, 0.0)


def seq2instance(data, P, Q):
    """
    将时间序列数据切分为监督学习所需的样本对 (X, Y)。
    - data: shape = (num_step, nodes, dims)
    - P: 输入序列长度
    - Q: 预测序列长度
    返回 x: (num_sample, P, nodes, dims), y: (num_sample, Q, nodes, dims)
    """
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y


def read_meta(path):
    """
    读取 metadata CSV（包含 Lat/Lng 列），返回 locations 数组，形状为 (2, N)：[lat, lng]
    """
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat,lng], 0)
    return locations


def construct_adj(data, num_node):
    """
    基于余弦相似度构造节点相似度矩阵（用于补齐 patch 时选择相似点）。
    方法：把时间序列按每天（24 小时 * 12 个采样点/小时）分块求均值，
    然后对所有节点间的均值向量计算 cosine similarity，最后做指数尺度变换以扩大差异。
    
    参数:
        data: shape (time_steps, nodes, 1) or (time_steps, nodes)
        num_node: 节点数量
    """
    # 按天切片并求均值，data.shape[0] 应为时间步数，假设每小时 12 个采样点
    num_days = data.shape[0] // (24 * 12)
    if num_days == 0:
        # 如果数据不足一天，直接对所有时间步求平均
        data_mean = np.mean(data, axis=0)  # (nodes, 1) or (nodes,)
    else:
        # 先对每天内的时间步求平均，再对不同天求平均
        daily_means = []
        for i in range(num_days):
            day_data = data[24*12*i: 24*12*(i+1)]  # (288, nodes, 1)
            day_mean = np.mean(day_data, axis=0)   # (nodes, 1)
            daily_means.append(day_mean)
        data_mean = np.mean(daily_means, axis=0)   # (nodes, 1)

    # 确保形状为 (nodes, features)
    data_mean = data_mean.squeeze()  # 移除所有长度为1的维度

    # 极端情况兜底：如果 squeeze 后成为标量，说明样本极少，退化为单位相似度矩阵
    if data_mean.ndim == 0:
        return np.eye(num_node, dtype=float)

    if data_mean.ndim == 1:
        data_mean = data_mean.reshape(-1, 1)  # (nodes, 1)
    elif data_mean.ndim > 2:
        # 如果还有多余维度，强制reshape
        data_mean = data_mean.reshape(num_node, -1)  # (nodes, features)

    # 如果节点维度与 num_node 不一致，进行裁剪或 padding
    if data_mean.shape[0] != num_node:
        if data_mean.shape[0] > num_node:
            data_mean = data_mean[:num_node]
        else:
            pad_rows = np.repeat(data_mean[-1:], num_node - data_mean.shape[0], axis=0)
            data_mean = np.concatenate([data_mean, pad_rows], axis=0)
    
    # 转置为 (nodes, time_features)，然后计算节点间的余弦相似度
    tem_matrix = cosine_similarity(data_mean, data_mean)  # (nodes, nodes)
    
    # 指数化并标准化，增强相似度的差别
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix


def augmentAlign(dist_matrix, auglen):
    """
    从 dist_matrix 中找到最相似的 auglen 个索引（去重），用于给小片段补齐相似节点。
    dist_matrix: 距离/相似度矩阵，函数会展平成一维后排序并映射回列索引
    """
    # 先按相似度降序排列索引（reshape 并乘 -1 再 argsort 实现降序）
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    # 由于 reshape 后的索引需要映射回原矩阵的列索引，取模映射
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    # 逐个选择不重复的索引直到凑够 auglen 个
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)


def reorderData(parts_idx, mxlen, adj, sps):
    """
    根据 kd-tree 划分的 parts_idx（每个叶子节点的点索引列表）进行重排与补齐：
    - ori_parts_idx: 原始点索引按 patch 顺序拼接
    - reo_parts_idx: 重排后的 patch 内相对索引（用于后续重排映射）
    - reo_all_idx: 对每个 patch 进行 padding 补齐后得到的全索引（包含原始与补齐点）
    参数说明：
      parts_idx: list of ndarray，每个元素是该叶子节点包含的原始点索引
      mxlen: 叶子节点内最大长度（未严格使用，但由 kdTree 返回）
      adj: 相似度矩阵，用于选择补齐点
      sps: 目标的每个小 patch 大小（若某个叶子节点小于 sps，则需要补齐）
    """
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        # 选出该 part 在全局 adj 中对应行（用于寻找相似点）
        part_dist = adj[part_idx, :].copy()
        # 把与自身的相似度置为 0，避免自己被选为补齐点
        part_dist[:, part_idx] = 0
        if sps-part_idx.shape[0] > 0:
            # 需要补齐：选择最相似的若干点
            local_part_idx = augmentAlign(part_dist, sps-part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        # reo_parts_idx 存储 patch 内相对位置的全局偏移（用于重排）
        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0])+sps*i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    # 防御性处理：确保返回的索引在合法范围内，避免后续索引越界（尤其是极小样本场景）
    try:
        num_node = adj.shape[0]
    except Exception:
        num_node = max(1, int(np.max(ori_parts_idx)) + 1 if ori_parts_idx.size > 0 else 1)

    if ori_parts_idx.size > 0:
        ori_parts_idx = np.clip(ori_parts_idx, 0, max(0, num_node - 1)).astype(int)
    if reo_parts_idx.size > 0:
        reo_parts_idx = reo_parts_idx.astype(int)
    if reo_all_idx.size > 0:
        reo_all_idx = np.clip(reo_all_idx, 0, max(0, num_node - 1)).astype(int)

    return ori_parts_idx, reo_parts_idx, reo_all_idx


def kdTree(locations, times, axis):
    """
    递归构造 kd-tree 的叶子节点划分：
    - locations: shape (2, N) 包含 [lat, lng]
    - times: 递归深度（切分次数），每次把当前点集均分为左右两部分
    - axis: 决定按第 0 维（lat）还是第 1 维（lng）进行排序分割，axis 会在递归中切换
    返回 parts: list of ndarrays（每个元素为该叶子节点的原始索引）, 和叶子最大长度
    """
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1]//2]), np.sort(sorted_idx[locations.shape[1]//2:])
    parts = []
    if times == 1:
        # 递归终止：返回两个叶子节点
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    else:
        # 继续对左右子集递归划分，axis^1 切换维度
        left_parts, lmxlen = kdTree(locations[:,part1], times-1, axis^1)
        right_parts, rmxlen = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)


def loadData(filepath, metapath, P, Q, train_ratio, test_ratio, adjpath, recurtimes, tod, dow, sps, log):
    """
    数据加载与预处理主函数：
    - 读取 traffic npz（假设 key 为 'data'），只取第一个通道（...,:1）
    - 读取元数据位置文件（lat/lng）
    - 构造时间特征 TE（time-of-day, day-of-week），并扩展到每个节点
    - 划分 train/val/test
    - 加载或构造相似度矩阵 adj
    - 使用 kdTree/ reorderData 进行空间划分与补齐索引（用于 patching）
    - 使用 seq2instance 将时间序列切分成样本对
    - 计算训练集的 mean/std 作为归一化参数
    返回大量预处理后的数组与索引，供训练/评估使用
    """
    # Traffic: 读取数据，npz 中的 'data'，只取到最后一维的第一个通道
    Traffic = np.load(filepath)['data'][...,:1]
    # 读取元数据（节点经纬度）
    locations = read_meta(metapath)
    num_step = Traffic.shape[0]
    # temporal positions: 构造时间特征 TE（两个维度：tod, dow）
    TE = np.zeros([num_step, 2])
    TE[:,0] = np.array([i % tod for i in range(num_step)])
    TE[:,1] = np.array([(i // tod) % dow for i in range(num_step)])
    # 将时间特征扩展到每个节点：shape -> (num_step, num_nodes, 2)
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)
    log_string(log, f'Shape of data: {Traffic.shape}')
    log_string(log, f'Shape of locations: {locations.shape}')
    # train/val/test 划分（按时间维度简单切分）
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    trainData, trainTE = Traffic[: train_steps], TE_tile[: train_steps]
    valData, valTE = Traffic[train_steps : train_steps + val_steps], TE_tile[train_steps : train_steps + val_steps]
    testData, testTE = Traffic[-test_steps :], TE_tile[-test_steps :]
    # load adj for padding（如果存在文件则直接加载，否则基于 trainData 计算并保存）
    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        adj = construct_adj(trainData, locations.shape[1])
        np.save(adjpath, adj)
    # partition and pad data with new indices（kdTree 划分并重排/补齐）
    parts_idx, mxlen = kdTree(locations, recurtimes, 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, sps)
    # X, Y 切分
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)
    # normalization: 使用训练集计算均值与标准差
    mean, std = np.mean(trainX), np.std(trainX)
    # log 一些关键信息
    log_string(log, f'Shape of Train: {trainY.shape}')
    log_string(log, f'Shape of Validation: {valY.shape}')
    log_string(log, f'Shape of Test: {testY.shape}')
    log_string(log, f'Mean: {mean} & Std: {std}')
    
    return trainX, trainY, trainXTE, trainYTE, valX, valY, valXTE, valYTE, testX, testY, testXTE, testYTE, mean, std, ori_parts_idx, reo_parts_idx, reo_all_idx
    

def loadDataFromSamples(sample_data_dict, locations, adjpath, recurtimes, sps, log):
    """从 PatchSTGSampleLoader 预处理的样本数据中构建训练/验证/测试集"""

    trainX = sample_data_dict['train']['X']
    trainY = sample_data_dict['train']['Y']
    trainTE = sample_data_dict['train']['TE']

    valX = sample_data_dict['val']['X']
    valY = sample_data_dict['val']['Y']
    valTE = sample_data_dict['val']['TE']

    testX = sample_data_dict['test']['X']
    testY = sample_data_dict['test']['Y']
    testTE = sample_data_dict['test']['TE']

    metadata = sample_data_dict['metadata']
    mean = metadata['mean']
    std = metadata['std']
    num_node = trainX.shape[2]

    log_string(log, f'Shape of trainX: {trainX.shape}')
    log_string(log, f'Shape of trainTE: {trainTE.shape}')
    log_string(log, f'Shape of locations: {locations.shape}')
    log_string(log, f'Number of nodes: {num_node}')
    log_string(log, f'Mean: {mean} & Std: {std}')

    if os.path.exists(adjpath):
        adj = np.load(adjpath)
        # 如果已有 adj 与当前数据的节点数不匹配则重建，避免使用历史大图导致索引越界
        if adj.shape[0] == num_node and adj.shape[1] == num_node:
            log_string(log, f'Loaded adj matrix from {adjpath}')
        else:
            log_string(
                log,
                f'WARNING: adj shape {adj.shape} != (num_node, num_node)=({num_node},{num_node}), will rebuild adj for current data.'
            )
            train_data_for_adj = trainX.transpose(1, 0, 2, 3).reshape(-1, num_node, 1)
            adj = construct_adj(train_data_for_adj, num_node)
            np.save(adjpath, adj)
            log_string(log, f'Constructed and saved adj matrix to {adjpath}')
    else:
        train_data_for_adj = trainX.transpose(1, 0, 2, 3).reshape(-1, num_node, 1)
        adj = construct_adj(train_data_for_adj, num_node)
        np.save(adjpath, adj)
        log_string(log, f'Constructed and saved adj matrix to {adjpath}')

    if locations.shape[1] != num_node:
        log_string(log, f'WARNING: locations维度({locations.shape[1]}) != num_node({num_node})')
        log_string(log, f'使用前{num_node}个位置进行KDTree划分')
        locations = locations[:, :num_node]

    parts_idx, mxlen = kdTree(locations, recurtimes, 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, sps)

    log_string(log, f'KDTree划分: {len(parts_idx)} 个patches, 最大长度={mxlen}')
    log_string(log, f'Reordered indices: ori={len(ori_parts_idx)}, reo={len(reo_parts_idx)}, all={len(reo_all_idx)}')

    trainYTE = trainTE.copy()
    valYTE = valTE.copy()
    testYTE = testTE.copy()

    log_string(log, f'Shape of Train: {trainY.shape}')
    log_string(log, f'Shape of Validation: {valY.shape}')
    log_string(log, f'Shape of Test: {testY.shape}')

    return trainX, trainY, trainTE, trainYTE, valX, valY, valTE, valYTE, testX, testY, testTE, testYTE, mean, std, ori_parts_idx, reo_parts_idx, reo_all_idx

