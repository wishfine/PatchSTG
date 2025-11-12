# 🚀 服务器部署与运行完整指南

这份指南将告诉你如何在远程服务器上从零开始运行整个 PatchSTG 项目（使用 ODPS 数据训练北京交通流量预测模型）。

---

## 📋 目录

1. [准备工作](#1-准备工作)
2. [克隆项目](#2-克隆项目)
3. [配置环境](#3-配置环境)
4. [设置 ODPS 凭证](#4-设置-odps-凭证)
5. [运行流程](#5-运行流程)
6. [监控训练](#6-监控训练)
7. [常见问题](#7-常见问题)

---

## 1. 准备工作

### 1.1 服务器要求

- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+)
- **Python**: 3.8 或更高版本
- **GPU**: 推荐（CUDA 11.x）, 但 CPU 也可以
- **内存**: 至少 16GB RAM
- **存储**: 至少 10GB 可用空间

### 1.2 检查 Python 版本

```bash
python --version
# 或
python3 --version
```

如果版本过低，建议使用 conda 安装新版本：

```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建新环境
conda create -n patchstg python=3.8
conda activate patchstg
```

---

## 2. 克隆项目

```bash
# 进入工作目录
cd ~  # 或你想要的目录

# 克隆项目
git clone https://github.com/wishfine/PatchSTG.git

# 进入项目目录
cd PatchSTG
```

---

## 3. 配置环境

### 3.1 安装依赖包

```bash
# 方法 A: 使用 pip 直接安装
pip install torch==1.11.0 pyodps pandas numpy tqdm scikit-learn jupyter

# 方法 B: 如果有 requirements.txt (可选)
# pip install -r requirements.txt
```

### 3.2 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import odps; print('PyODPS: OK')"
python -c "import pandas; print('Pandas: OK')"
```

如果没有报错，说明依赖安装成功。

---

## 4. 设置 ODPS 凭证

⚠️ **重要**: 需要有效的阿里云 MaxCompute 凭证才能访问数据。

### 4.1 永久设置（推荐）

编辑你的 shell 配置文件：

```bash
# 如果使用 bash
vim ~/.bashrc

# 如果使用 zsh
vim ~/.zshrc
```

在文件末尾添加：

```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID='你的_ACCESS_KEY_ID'
export ALIBABA_CLOUD_ACCESS_KEY_SECRET='你的_ACCESS_KEY_SECRET'
```

保存后重新加载：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 4.2 临时设置（当前会话）

```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID='你的_ACCESS_KEY_ID'
export ALIBABA_CLOUD_ACCESS_KEY_SECRET='你的_ACCESS_KEY_SECRET'
```

### 4.3 验证凭证

```bash
python -c "import os; print('ID:', os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID')[:10] + '...')"
```

---

## 5. 运行流程

### 5.1 使用 Jupyter Notebook（推荐）

如果你想交互式地运行和监控，使用 Jupyter Notebook 是最佳选择。

#### Step 1: 启动 Jupyter

```bash
# 在服务器上启动 Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# 你会看到类似这样的输出：
# [I 15:30:00.000 NotebookApp] http://hostname:8888/?token=abc123...
```

#### Step 2: 本地访问 Jupyter

在你的本地电脑上建立 SSH 隧道：

```bash
# 在本地终端运行
ssh -L 8888:localhost:8888 username@server_ip

# 然后在浏览器打开
http://localhost:8888
# 输入 token（从服务器输出中复制）
```

#### Step 3: 按顺序运行 Notebooks

**① `quickstart_odps.ipynb` - 环境检查（约 1 分钟）**

```
目的: 检查环境是否配置正确
检查内容:
  ✅ ODPS 凭证
  ✅ Python 依赖
  ✅ 项目目录结构
  
预期输出: 所有检查项都显示 ✅
```

**② `test_location_loading.ipynb` - 验证元数据（约 2-5 分钟）**

```
目的: 验证元数据表中的位置信息
检查内容:
  ✅ 表结构正确
  ✅ 位置覆盖率 >= 50%
  ✅ 位置范围合理
  
预期输出:
  城市: 北京 (adcode=110000)
  节点数: ~15,000 个
  位置覆盖率: 100%
```

**③ `test_dense_data_loader.ipynb` - 验证数据格式（约 5-10 分钟）**

```
目的: 验证数据加载逻辑和格式
检查内容:
  ✅ 数据形状正确 (samples, 12, nodes, 1)
  ✅ 数据密集性 > 50%
  ✅ 无 NaN/Inf 值
  ✅ 空间 patch 创建成功
  
预期输出:
  训练集: ~8,000 样本
  验证集: ~2,000 样本
  测试集: ~2,000 样本
  数据密集性: 70-80%
```

**④ `train_model.ipynb` - 训练模型（预计 2-8 小时）**

```
目的: 训练 PatchSTG 模型
配置:
  城市: 北京
  日期: 7 天数据
  Epochs: 50
  Batch Size: 32
  
训练过程:
  Epoch 1/50: Loss=xxx.xxxx, MAE=xx.xx, Time=xx.xs
  Epoch 2/50: Loss=xxx.xxxx, MAE=xx.xx, Time=xx.xs
  ...
  ✅ 保存最佳模型 (Epoch xx, MAE=xx.xx)
  
输出文件:
  模型: ./saved_models/odps_beijing_model.pth
  日志: ./log/odps_beijing_log
```

---

### 5.2 使用 Python 脚本（命令行方式）

如果你更喜欢命令行方式：

#### Step 1: 环境检查

```bash
python -c "
import os
import sys
print('Python:', sys.version)
print('ODPS Key:', 'SET' if os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID') else 'NOT SET')
"
```

#### Step 2: 检查配置文件

```bash
cat config/ODPS.conf | grep -E "adcode|start_date|end_date"
```

确认：
- `adcode = 110000` (北京)
- `start_date = 20250919`
- `end_date = 20250925`

#### Step 3: 运行训练

```bash
# 创建日志目录
mkdir -p log saved_models

# 开始训练（会运行很长时间）
nohup python train_odps.py --config config/ODPS.conf --mode train > train.log 2>&1 &

# 记录进程 ID
echo $! > train.pid
```

---

## 6. 监控训练

### 6.1 查看实时日志

```bash
# 方法 1: 查看训练输出日志
tail -f train.log

# 方法 2: 查看模型日志文件
tail -f ./log/odps_beijing_log

# 按 Ctrl+C 退出查看
```

### 6.2 检查训练进度

```bash
# 查看最近的日志条目
tail -20 ./log/odps_beijing_log

# 搜索最佳模型信息
grep "保存最佳模型" ./log/odps_beijing_log
```

### 6.3 检查进程状态

```bash
# 检查训练进程是否还在运行
ps aux | grep train_odps.py

# 或使用 PID 文件
cat train.pid
ps -p $(cat train.pid)
```

### 6.4 检查 GPU 使用（如果有 GPU）

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看 GPU 内存使用
nvidia-smi --query-gpu=memory.used --format=csv
```

---

## 7. 常见问题

### Q1: 如何停止训练？

```bash
# 找到进程 ID
ps aux | grep train_odps.py

# 或从文件读取
cat train.pid

# 优雅停止（推荐）
kill $(cat train.pid)

# 强制停止（如果上面不行）
kill -9 $(cat train.pid)
```

### Q2: 训练中断了怎么办？

训练会自动保存最佳模型，可以从断点继续：

```python
# 在 train_model.ipynb 或 train_odps.py 中
# 模型会自动从 saved_models/odps_beijing_model.pth 加载
```

### Q3: 内存不足错误

```bash
# 减小 batch size
vim config/ODPS.conf
# 修改: batch_size = 16  (从 32 改为 16)
```

### Q4: 数据加载太慢

```bash
# 增加 num_workers
vim lib/odps_data_loader.py
# 找到 DataLoader，增加 num_workers
```

### Q5: 如何更改训练城市？

```bash
vim config/ODPS.conf

# 修改城市代码：
# 北京: 110000
# 上海: 310000
# 广州: 440100
# 深圳: 440300
# 杭州: 330100
```

### Q6: 如何查看模型参数？

```python
import torch
model = torch.load('./saved_models/odps_beijing_model.pth')
print(f"模型大小: {sum(p.numel() for p in model.values())} 参数")
```

### Q7: 训练完成后如何测试？

```bash
# 使用 Jupyter Notebook
# 运行 train_model.ipynb 的测试部分（最后几个单元格）

# 或使用 Python 脚本
python train_odps.py --config config/ODPS.conf --mode test
```

---

## 📊 预期结果

训练完成后，你应该看到：

```
✅ 文件生成:
   ./saved_models/odps_beijing_model.pth  (约 50-100 MB)
   ./log/odps_beijing_log                 (训练日志)
   ./training_curves.png                  (可选: 训练曲线图)

✅ 测试指标:
   MAE:  5-10 (取决于数据质量)
   RMSE: 8-15
   MAPE: 15-25%
```

---

## 🎯 完整流程时间线

| 步骤 | 时间 | 说明 |
|------|------|------|
| 1. 克隆项目 | 1 分钟 | `git clone` |
| 2. 安装依赖 | 5-10 分钟 | `pip install` |
| 3. 设置凭证 | 2 分钟 | 编辑 ~/.bashrc |
| 4. 环境检查 | 1 分钟 | `quickstart_odps.ipynb` |
| 5. 元数据验证 | 2-5 分钟 | `test_location_loading.ipynb` |
| 6. 数据格式验证 | 5-10 分钟 | `test_dense_data_loader.ipynb` |
| 7. 模型训练 | 2-8 小时 | `train_model.ipynb` |
| 8. 模型测试 | 5-10 分钟 | `train_model.ipynb` 最后部分 |

**总计**: ~3-9 小时（主要取决于训练时间）

---

## 🔗 相关文档

- `README.md` - 项目总览
- `JUPYTER_NOTEBOOKS_GUIDE.md` - Jupyter 使用详细指南
- `DATA_PIPELINE_FLOW.md` - 数据流程说明
- `config/ODPS.conf` - 配置文件

---

## 💡 最佳实践

1. **首次运行**: 使用 Jupyter Notebook，可以看到详细输出和进度
2. **长时间训练**: 使用 `nohup` + Python 脚本，避免 SSH 断开
3. **调试问题**: 在 Jupyter 中交互式运行，逐步检查中间结果
4. **生产环境**: 使用 Python 脚本 + tmux/screen 会话

---

## 🆘 需要帮助？

1. 查看日志文件: `./log/odps_beijing_log`
2. 检查错误输出: `train.log`
3. 在 Jupyter 中交互式调试
4. 查看详细文档（上面列出的 Markdown 文件）

---

**祝训练顺利！🚀**
