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

# 引入模型与工具函数
from models.model import PatchSTG
from lib.utils import log_string, loadData, _compute_loss, metric

class Solver(object):
    # 默认配置占位（可以被传入的 config 覆盖）
    DEFAULTS = {}

    def __init__(self, config):
        """
        Solver 构造函数：把传入的 config 字典扩展到 self 上，
        然后加载数据、初始化模型与优化器等。

        参数:
            config (dict): 从 argparse/配置文件解析出的参数字典
        """
        # 把 DEFAULTS 与外部传入的配置合并到实例属性中
        self.__dict__.update(Solver.DEFAULTS, **config)

        # 记录日志：开始加载数据
        log_string(log, '\n------------ Loading Data -------------')
        # 下面是 loadData 返回的多个值的说明：
        # - trainX/trainY/..: 划分后的训练/验证/测试数据与对应时间特征
        # - mean/std: 训练数据的均值与标准差（用于归一化/反归一化）
        # - ori_parts_idx/reo_parts_idx/reo_all_idx: kd-tree 划分与重排索引，用于 patching
        # 具体细节在 lib/utils.py 的 loadData 函数中实现
        self.trainX, self.trainY, self.trainXTE, self.trainYTE,\
        self.valX, self.valY, self.valXTE, self.valYTE,\
        self.testX, self.testY, self.testXTE, self.testYTE,\
        self.mean, self.std,\
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = loadData(
                                        self.traffic_file, self.meta_file,
                                        self.input_len, self.output_len,
                                        self.train_ratio, self.test_ratio,
                                        self.adj_file, self.recur_times,
                                        self.tod, self.dow,
                                        self.spa_patchsize, log)
        log_string(log, '------------ End -------------\n')

        # 训练过程中的最佳 epoch（用于保存最优模型）
        self.best_epoch = 0

        # 设备选择：优先使用指定的 GPU，否则使用 CPU
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        # 构建模型与优化器
        self.build_model()
    
    def build_model(self):
        """
        构建模型、优化器与学习率调度器。
        - PatchSTG 的构造函数参数来自配置：时间/空间 patch、节点数、嵌入维度等。
        - 模型被移动到之前选择的 device 上（CPU/GPU）。
        """
        # 实例化模型并放到 device
        self.model = PatchSTG(self.output_len, self.tem_patchsize, self.tem_patchnum,
                            self.node_num, self.spa_patchsize, self.spa_patchnum,
                            self.tod, self.dow,
                            self.layers, self.factors,
                            self.input_dims, self.node_dims, self.tod_dims, self.dow_dims,
                            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx).to(self.device)

        # 优化器：AdamW（包含 weight_decay 正则项）
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,weight_decay=self.weight_decay)

        # 学习率调度器：在指定的里程碑处衰减学习率
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1,35,40],
            gamma=0.5,
        )

    def vali(self):
        # 验证模式：关闭 dropout 等训练相关行为
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []  # 用于收集分批预测值（反标准化后）
        label = [] # 用于收集真实标签

        # 根据 batch_size 计算需要的 batch 数
        num_batch = math.ceil(num_val / self.batch_size)
        # 验证时不需要计算梯度，节省显存与计算
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    # 计算当前批次的样本切片范围
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    # 从 numpy 缓存中切片出 X/Y/TE
                    X = self.valX[start_idx : end_idx]
                    Y = self.valY[start_idx : end_idx]
                    # TE 与模型在同一 device 上
                    TE = torch.from_numpy(self.valXTE[start_idx : end_idx]).to(self.device)
                    # 归一化输入（使用训练集的均值与方差）并转为 float tensor
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    # 前向推理
                    y_hat = self.model(NormX,TE)

                    # 反标准化并转回 numpy，用于后续指标计算
                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        # 把所有分批结果拼接为完整数组
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        # 逐步计算每个时间步的指标（MAE, RMSE, MAPE），并记录
        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        # 计算整体（所有预测 horizon）的平均指标并返回
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        # 训练模式入口
        log_string(log, "======================TRAIN MODE======================")
        # 用一个足够大的初始最小 loss 方便后续比较
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]

        # 迭代多个 epoch
        for epoch in tqdm(range(1,self.max_epoch+1)):
            # 进入训练模式（启用 dropout 等）
            self.model.train()
            train_l_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0, time.time()

            # 随机打乱训练样本（按样本维度打乱，保持时间序列内部完整）
            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainXTE = self.trainXTE[permutation]

            num_batch = math.ceil(num_train / self.batch_size)
            # tqdm 显示每个 epoch 的进度条
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    # 切片当前 batch 的数据
                    X = self.trainX[start_idx : end_idx]
                    Y = self.trainY[start_idx : end_idx]
                    TE = torch.from_numpy(self.trainXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)
                    
                    # 移动 Y 到 device 以便计算 loss
                    Y = torch.from_numpy(Y).float().to(self.device)
                    
                    # 梯度清零
                    self.optimizer.zero_grad()

                    # 前向
                    y_hat = self.model(NormX,TE)

                    # loss：把预测反标准化到原尺度再计算 masked mae
                    loss = _compute_loss(Y, y_hat*self.std+self.mean)
                    
                    # 反向传播与梯度剪裁
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    batch_count += 1
                    pbar.update(1)
            # 每个 epoch 结束后记录训练信息并在验证集上评估
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            mae, rmse, mape = self.vali()
            # 更新学习率
            self.lr_scheduler.step()
            # 保存最优模型（以最后一个 horizon 的 mae 作为判断）
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        # 测试入口：加载最优模型并在测试集上评估
        log_string(log, "======================TEST MODE======================")
        # 从磁盘加载模型参数（在 CPU/GPU 之间切换由 map_location 控制）
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.testX[start_idx : end_idx]
                    Y = self.testY[start_idx : end_idx]
                    TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    y_hat = self.model(NormX,TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    # -----------------------
    # 命令行参数定义（从配置文件中读取默认值）
    # -----------------------
    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])

    # 数据相关参数
    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    # 模型结构与 patch 参数（来自 config/param）
    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['tps'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['tpn'])
    parser.add_argument('--factors', type=int, default = config['param']['factors'])
    parser.add_argument('--recur_times', type = int, default = config['param']['recur'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['sps'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['spn'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['id'])
    parser.add_argument('--node_dims', type=int, default = config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dd'])

    # 文件路径参数
    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])

    # 再次解析命令行（这会把上面添加的参数解析为最终 args）
    args = parser.parse_args()

    # 打开日志文件（以写模式覆盖），log 在程序其它地方被传入记录信息
    log = open(args.log_file, 'w')

    # 随机种子设置，保证可复现（在多平台上可能仍有微差异）
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # 打印/记录所有的运行选项到日志中，便于复现与调试
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    # 使用解析后的参数 dict 创建 Solver 实例
    solver = Solver(vars(args))

    # 默认执行测试流程（如果你想训练，把上一行注释，打开下面一行）
    # solver.train()
    solver.test()
    
