"""
使用 ODPS 数据进行训练
从 MaxCompute 表加载数据并训练 PatchSTG 模型
"""
import math
import time
import torch
import torch.nn as nn
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm

# 引入模型与工具函数
from models.model import PatchSTG
from lib.utils import log_string, _compute_loss, metric
from lib.odps_data_loader import ODPSDataLoader


class ODPSSolver(object):
    """使用 ODPS 数据的 Solver"""
    
    DEFAULTS = {}

    def __init__(self, config, data_loader=None):
        """
        构造函数
        
        参数:
            config (dict): 配置字典
            data_loader (ODPSDataLoader): 可选的预加载数据加载器
        """
        self.__dict__.update(ODPSSolver.DEFAULTS, **config)

        # 初始化或使用提供的数据加载器
        if data_loader is None:
            self.data_loader = ODPSDataLoader(config, log)
            self.data_loader.load_data()
        else:
            self.data_loader = data_loader
            if not self.data_loader._loaded:
                self.data_loader.load_data()
        
        # 从数据加载器获取数据
        self.trainX, self.trainY, self.trainXTE, self.trainYTE = self.data_loader.get_train_data()
        self.valX, self.valY, self.valXTE, self.valYTE = self.data_loader.get_val_data()
        self.testX, self.testY, self.testXTE, self.testYTE = self.data_loader.get_test_data()
        self.mean, self.std = self.data_loader.get_normalization_params()
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = self.data_loader.get_patch_indices()
        
        # 从数据中获取实际的节点数
        self.node_num = self.data_loader.node_num
        
        self.best_epoch = 0
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()
    
    def build_model(self):
        """构建模型、优化器与学习率调度器"""
        self.model = PatchSTG(
            self.output_len, self.tem_patchsize, self.tem_patchnum,
            self.node_num, self.spa_patchsize, self.spa_patchnum,
            self.tod, self.dow,
            self.layers, self.factors,
            self.input_dims, self.node_dims, self.tod_dims, self.dow_dims,
            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1, 35, 40],
            gamma=0.5,
        )

    def vali(self):
        """验证"""
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                X = self.valX[start_idx:end_idx]
                Y = self.valY[start_idx:end_idx]
                TE = torch.from_numpy(self.valXTE[start_idx:end_idx]).to(self.device)
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)

                y_hat = self.model(NormX, TE)

                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
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
            log_string(log, 'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        """训练"""
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]

        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0, time.time()

            # 打乱训练数据
            self.data_loader.shuffle_train_data()
            self.trainX, self.trainY, self.trainXTE, self.trainYTE = self.data_loader.get_train_data()

            num_batch = math.ceil(num_train / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    X = self.trainX[start_idx:end_idx]
                    Y = self.trainY[start_idx:end_idx]
                    TE = torch.from_numpy(self.trainXTE[start_idx:end_idx]).to(self.device)
                    NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)
                    Y = torch.from_numpy(Y).float().to(self.device)
                    
                    self.optimizer.zero_grad()
                    y_hat = self.model(NormX, TE)
                    loss = _compute_loss(Y, y_hat * self.std + self.mean)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()
                    batch_count += 1
                    pbar.update(1)
            
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], 
                   train_l_sum / batch_count, time.time() - start))
            
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step()
            
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        """测试"""
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_test = self.testX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_test / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_test, (batch_idx + 1) * self.batch_size)

                X = self.testX[start_idx:end_idx]
                Y = self.testY[start_idx:end_idx]
                TE = torch.from_numpy(self.testXTE[start_idx:end_idx]).to(self.device)
                NormX = torch.from_numpy((X - self.mean) / self.std).float().to(self.device)

                y_hat = self.model(NormX, TE)

                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
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
            log_string(log, 'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ODPS.conf", 
                       help='configuration file')
    parser.add_argument("--mode", type=str, default="train", 
                       choices=['train', 'test', 'both'],
                       help='run mode: train, test, or both')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # 训练参数
    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
    parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default=config['train']['weight_decay'])

    # ODPS 数据参数
    parser.add_argument('--odps_project', type=str, default=config['data']['odps_project'])
    parser.add_argument('--odps_endpoint', type=str, default=config['data']['odps_endpoint'])
    parser.add_argument('--odps_table', type=str, default=config['data']['odps_table'])
    parser.add_argument('--adcode', type=str, default=config['data']['adcode'])
    parser.add_argument('--start_date', type=str, default=config['data']['start_date'])
    parser.add_argument('--end_date', type=str, default=config['data']['end_date'])
    parser.add_argument('--input_len', type=int, default=config['data']['input_len'])
    parser.add_argument('--output_len', type=int, default=config['data']['output_len'])
    parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])

    # 模型参数
    parser.add_argument('--layers', type=int, default=config['param']['layers'])
    parser.add_argument('--tem_patchsize', type=int, default=config['param']['tps'])
    parser.add_argument('--tem_patchnum', type=int, default=config['param']['tpn'])
    parser.add_argument('--factors', type=int, default=config['param']['factors'])
    parser.add_argument('--recur_times', type=int, default=config['param']['recur'])
    parser.add_argument('--spa_patchsize', type=int, default=config['param']['sps'])
    parser.add_argument('--spa_patchnum', type=int, default=config['param']['spn'])
    parser.add_argument('--node_num', type=int, default=config['param']['nodes'])
    parser.add_argument('--tod', type=int, default=config['param']['tod'])
    parser.add_argument('--dow', type=int, default=config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default=config['param']['id'])
    parser.add_argument('--node_dims', type=int, default=config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default=config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default=config['param']['dd'])

    # 文件路径
    parser.add_argument('--model_file', default=config['file']['model'])
    parser.add_argument('--log_file', default=config['file']['log'])

    args = parser.parse_args()
    log = open(args.log_file, 'w')

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # 打印配置
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    # 创建 solver
    solver = ODPSSolver(vars(args))

    # 根据模式运行
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:  # both
        solver.train()
        solver.test()
    
    log.close()
