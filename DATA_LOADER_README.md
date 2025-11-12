# 数据加载与预处理模块使用说明

## 概述

本项目已将数据加载和预处理逻辑从训练流程中分离，提供了更灵活的数据管理方式。

## 文件结构

```
lib/
  ├── data_loader.py         # 数据加载器类
  └── utils.py               # 原有的工具函数
  
preprocess_data.py           # 独立的数据预处理脚本
train_with_preloaded_data.py # 使用预加载数据训练的示例
main.py                      # 主训练脚本（已更新）
```

## 使用方式

### 方式 1: 独立预处理数据

在训练之前，可以先预处理和检查数据：

```bash
# 预处理数据并生成数据信息文件
python preprocess_data.py --config config/CA.conf --output data_info.json
```

这将会：
- 加载和预处理数据
- 生成数据统计信息（样本数、节点数、归一化参数等）
- 保存到 JSON 文件
- 生成预处理日志

### 方式 2: 使用预加载数据训练

使用独立的训练脚本，明确展示数据加载和训练的分离：

```bash
# 使用预加载数据进行训练
python train_with_preloaded_data.py --config config/CA.conf
```

这个脚本展示了：
1. 先加载数据（Step 1）
2. 再训练模型（Step 2）
3. 最后测试模型（Step 3）

### 方式 3: 原有方式（已更新为使用 DataLoader）

原有的 `main.py` 已经更新为使用 `DataLoader`，但保持了相同的接口：

```bash
# 直接运行（内部自动使用 DataLoader）
python main.py --config config/CA.conf
```

## DataLoader 类的主要功能

### 初始化和加载
```python
from lib.data_loader import DataLoader

# 创建数据加载器
data_loader = DataLoader(config_dict, log_file)

# 加载数据
data_loader.load_data()
```

### 获取数据
```python
# 获取训练数据
trainX, trainY, trainXTE, trainYTE = data_loader.get_train_data()

# 获取验证数据
valX, valY, valXTE, valYTE = data_loader.get_val_data()

# 获取测试数据
testX, testY, testXTE, testYTE = data_loader.get_test_data()

# 获取归一化参数
mean, std = data_loader.get_normalization_params()

# 获取 patch 索引
ori_parts_idx, reo_parts_idx, reo_all_idx = data_loader.get_patch_indices()
```

### 数据操作
```python
# 打乱训练数据
data_loader.shuffle_train_data(seed=42)

# 归一化
normalized_data = data_loader.normalize_data(raw_data)

# 反归一化
original_data = data_loader.denormalize_data(normalized_data)

# 获取数据信息
info = data_loader.get_data_info()
# 返回: {'train_samples': 1000, 'val_samples': 200, ...}
```

## 优势

### 1. 职责分离
- **数据加载**: `DataLoader` 类专门负责数据处理
- **模型训练**: `Solver` 类专注于模型训练和评估
- **解耦合**: 可以独立测试和优化每个模块

### 2. 代码复用
```python
# 可以在多个地方使用同一个数据加载器
data_loader = DataLoader(config, log)
data_loader.load_data()

# 用于训练
solver1 = Solver(config1, data_loader=data_loader)

# 用于不同配置的实验
solver2 = Solver(config2, data_loader=data_loader)
```

### 3. 便于调试
```python
# 单独检查数据
data_loader = DataLoader(config, log)
data_loader.load_data()

# 查看数据信息
print(data_loader.get_data_info())

# 检查数据形状
trainX, trainY, _, _ = data_loader.get_train_data()
print(f"Train X shape: {trainX.shape}")
print(f"Train Y shape: {trainY.shape}")
```

### 4. 更好的可测试性
```python
# 可以为数据加载编写单元测试
import unittest

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data_loader = DataLoader(test_config)
        data_loader.load_data()
        self.assertTrue(data_loader._loaded)
        
    def test_normalization(self):
        # 测试归一化/反归一化
        ...
```

## 配置要求

`DataLoader` 需要以下配置参数：

```python
config = {
    # 文件路径
    'traffic_file': 'path/to/traffic.npz',
    'meta_file': 'path/to/meta.csv',
    'adj_file': 'path/to/adj.pkl',
    
    # 数据参数
    'input_len': 12,
    'output_len': 12,
    'train_ratio': 0.7,
    'test_ratio': 0.2,
    
    # 模型参数
    'recur_times': 1,
    'tod': 1,
    'dow': 1,
    'spa_patchsize': 4,
}
```

## 迁移说明

如果你有使用旧版 `main.py` 的代码，只需要：

1. **导入新模块**:
```python
from lib.data_loader import DataLoader
```

2. **（可选）预加载数据**:
```python
# 在 Solver 之前加载
data_loader = DataLoader(config, log)
data_loader.load_data()

# 传给 Solver
solver = Solver(config, data_loader=data_loader)
```

3. **原有代码仍然可用**:
```python
# 这样仍然可以工作（内部自动创建 DataLoader）
solver = Solver(config)
```

## 示例工作流

### 研究场景：尝试不同的数据划分
```python
# 加载一次数据
base_config = {...}
data_loader = DataLoader(base_config, log)
data_loader.load_data()

# 实验 1: 正常训练
solver1 = Solver(config1, data_loader=data_loader)
solver1.train()

# 实验 2: 不同的超参数
solver2 = Solver(config2, data_loader=data_loader)
solver2.train()
```

### 生产场景：数据质量检查
```bash
# 第一步：检查数据
python preprocess_data.py --config config/CA.conf

# 查看生成的 data_info.json
cat data_info.json

# 第二步：确认无误后训练
python train_with_preloaded_data.py --config config/CA.conf
```

## 注意事项

1. **内存管理**: 数据加载器会将所有数据加载到内存中，确保有足够的 RAM
2. **数据一致性**: 多个 Solver 共享同一个 DataLoader 时，注意 `shuffle_train_data()` 会影响所有使用者
3. **日志文件**: 如果传入 `log` 参数，数据加载过程会被记录

## 常见问题

**Q: 为什么要分离数据加载？**  
A: 提高代码可维护性、便于测试、支持数据复用。

**Q: 会影响性能吗？**  
A: 不会。数据加载逻辑完全相同，只是组织方式更好。

**Q: 必须使用新方式吗？**  
A: 不必须。`main.py` 保持向后兼容，原有代码仍可运行。

**Q: 如何保存预处理后的数据？**  
A: 可以在 `DataLoader` 中添加 `save()` 和 `load()` 方法来序列化数据。
