# 项目修改总结 - 支持 ODPS 数据训练

## 修改概述

本次修改为 PatchSTG 项目添加了从 MaxCompute (ODPS) 表直接加载数据进行训练的功能，使得可以使用生产环境中的实时交通数据，而不仅限于预处理的 NPZ 文件。

## 新增文件

### 1. 核心数据加载模块

#### `lib/odps_data_loader.py`
- **功能**: ODPS 数据加载器类
- **主要特性**:
  - 连接 MaxCompute 并执行 SQL 查询
  - 解析 `time_feat` 和 `dym_feat_feat` 字符串字段
  - 自动构建节点列表（路段对）
  - 使用滑动窗口生成训练样本
  - 自动计算归一化参数
  - 支持数据集划分（train/val/test）

#### `lib/data_loader.py`
- **功能**: 通用数据加载器（用于 NPZ 文件）
- **主要特性**:
  - 封装原有的 `loadData` 函数
  - 提供统一的数据访问接口
  - 支持数据归一化/反归一化
  - 支持训练数据打乱

### 2. 训练脚本

#### `train_odps.py`
- **功能**: 使用 ODPS 数据训练的主脚本
- **主要特性**:
  - `ODPSSolver` 类，专门用于 ODPS 数据训练
  - 支持 train/test/both 三种运行模式
  - 从数据中自动获取节点数
  - 完全兼容原有的模型架构

### 3. 配置文件

#### `config/ODPS.conf`
- **功能**: ODPS 训练的配置文件模板
- **配置项**:
  - ODPS 连接参数（project, endpoint, table）
  - 数据过滤条件（adcode, start_date, end_date）
  - 序列长度（input_len, output_len）
  - 数据集划分比例
  - 模型超参数

### 4. 工具脚本

#### `quickstart_odps.py`
- **功能**: 快速开始检查脚本
- **检查项**:
  - ODPS 凭证配置
  - Python 依赖包安装
  - 必要目录创建
  - 提供下一步操作指引

#### `check_odps_data.py`
- **功能**: 数据质量检查脚本
- **输出内容**:
  - 数据统计信息（样本数、节点数）
  - 数据范围分析（最小值、最大值、均值等）
  - 零值比例检查
  - 生成 JSON 格式的数据报告

### 5. 文档

#### `ODPS_TRAINING_GUIDE.md`
- **内容**: 完整的 ODPS 训练指南
- **章节**:
  - 数据表结构说明
  - 环境配置步骤
  - 三种使用方式详解
  - 数据加载流程说明
  - 配置参数参考
  - 故障排除指南

#### `DATA_LOADER_README.md`
- **内容**: 数据加载器使用说明
- **章节**:
  - 文件结构说明
  - 三种使用方式
  - DataLoader 类功能详解
  - 优势与使用场景
  - 迁移指南

### 6. 更新的文件

#### `main.py`
- **修改**:
  - 导入 `DataLoader` 类
  - 修改 `Solver.__init__()` 接受可选的 `data_loader` 参数
  - 使用 `data_loader.shuffle_train_data()` 代替手动打乱
  - 保持向后兼容

#### `README.md`
- **新增**:
  - ODPS 快速开始部分
  - 城市代码参考表
  - 相关文档链接

## 数据流转换

### 原始 NPZ 数据格式
```
- 本地文件: data/CA/ca.npz
- 固定节点数: 预定义
- 空间关系: 邻接矩阵文件
- 时间特征: 预计算
```

### ODPS 表数据格式
```
- 远程表: autonavi_traffic_report.tb_inter_spatial_method_pretrain_data
- 动态节点: 从数据中提取 (nds_id, next_nds_id) 对
- 空间关系: 有向路段（可选构建邻接关系）
- 时间特征: 实时解析 time_feat 字符串
```

## 数据表结构

### 字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| nds_id | STRING | 起始路口节点 ID |
| next_nds_id | STRING | 终止路口节点 ID |
| adcode | STRING | 行政区划代码 |
| ds | STRING | 数据分区日期 |
| passts_time | DATETIME | 当前时刻 |
| flow_label | BIGINT | 目标流量值 |
| time_feat | STRING | 时间特征序列（24段） |
| dym_feat_feat | STRING | 动态流量序列（24个值） |

### time_feat 格式
```
"week hour minute day_type day month;week hour minute day_type day month;..."
```
- 共 24 段，分号分隔
- 每段 6 个维度，空格分隔
- 表示过去 24 分钟的时间特征

### dym_feat_feat 格式
```
"15;8;0;12;...;3"
```
- 共 24 个值，分号分隔
- 表示过去 24 分钟的实际流量

## 使用流程

### 1. 环境准备
```bash
# 设置凭证
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_key_id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_secret"

# 检查环境
python quickstart_odps.py
```

### 2. 配置调整
编辑 `config/ODPS.conf`:
```ini
adcode = 110000        # 选择城市
start_date = 20250701  # 设置日期范围
end_date = 20250731
```

### 3. 数据检查
```bash
python check_odps_data.py --config config/ODPS.conf
```

### 4. 开始训练
```bash
python train_odps.py --config config/ODPS.conf --mode train
```

## 关键实现细节

### 节点定义
- **NPZ 数据**: 单个交叉口作为节点
- **ODPS 数据**: (nds_id, next_nds_id) 对作为节点（有向路段）

### 样本生成
```python
# 对每个节点对，使用滑动窗口
for i in range(len(flow_series) - input_len - output_len + 1):
    x = flow_series[i:i+input_len]         # 输入序列
    y = flow_series[i+input_len:i+input_len+output_len]  # 输出序列
```

### 时间特征提取
```python
# 从 time_feat 的第一段提取当前时刻特征
time_feat = parse_time_feat(row['time_feat'])
tod = time_feat[0, 1] / 24.0  # hour / 24
dow = time_feat[0, 0] / 7.0   # week / 7
```

### 数据形状
```
X:   (num_samples, input_len, num_nodes, 1)
Y:   (num_samples, output_len, num_nodes, 1)
XTE: (num_samples, input_len, 2)  # [tod, dow]
YTE: (num_samples, output_len, 2)
```

## 优势

1. **实时性**: 直接从生产数据库查询最新数据
2. **灵活性**: 可以按城市、日期范围动态筛选
3. **可扩展性**: 易于添加新的特征字段
4. **一致性**: 与 NPZ 数据训练流程完全兼容
5. **可维护性**: 模块化设计，易于测试和调试

## 兼容性

### 向后兼容
- 原有的 `main.py` 仍然可用
- NPZ 数据训练方式不受影响
- 模型架构完全不变

### 新旧对比

| 特性 | NPZ 数据 | ODPS 数据 |
|------|----------|-----------|
| 数据源 | 本地文件 | 远程数据库 |
| 数据更新 | 手动 | 自动 |
| 城市选择 | 固定 | 动态 |
| 日期范围 | 固定 | 动态 |
| 节点数 | 固定 | 自动检测 |

## 注意事项

### 1. 内存管理
- ODPS 查询可能返回大量数据
- 建议限制日期范围（如一个月）
- 监控内存使用情况

### 2. 网络连接
- 需要稳定的网络连接到 ODPS
- 查询大数据集可能需要较长时间
- 建议先用小数据集测试

### 3. 凭证安全
- 不要将凭证硬编码在代码中
- 使用环境变量管理凭证
- 确保 .gitignore 包含凭证文件

### 4. 数据质量
- 运行 `check_odps_data.py` 检查数据
- 注意零值比例
- 检查时间序列连续性

## 测试建议

### 小规模测试
```bash
# 1. 使用短日期范围
start_date = 20250701
end_date = 20250703  # 只用 3 天数据

# 2. 减少训练轮数
max_epoch = 5

# 3. 使用小 batch size
batch_size = 16
```

### 验证流程
1. 检查环境: `python quickstart_odps.py`
2. 检查数据: `python check_odps_data.py`
3. 小规模训练: 3天数据，5个epoch
4. 验证指标: 与 NPZ 数据结果对比
5. 扩大规模: 逐步增加日期范围和训练轮数

## 常见问题

### Q1: 如何选择合适的日期范围？
**A**: 
- 开始: 1周数据（快速测试）
- 正常: 1个月数据（平衡训练效果和时间）
- 充分: 3个月数据（最佳效果）

### Q2: 节点数太多怎么办？
**A**: 
- 选择特定城市（使用 adcode）
- 缩短日期范围
- 考虑采样策略

### Q3: 训练很慢怎么办？
**A**: 
- 减少 batch_size
- 减少 max_epoch
- 使用 GPU
- 减少数据量

### Q4: 如何验证数据正确性？
**A**: 
```bash
# 1. 运行数据检查
python check_odps_data.py --config config/ODPS.conf

# 2. 查看生成的 JSON 报告
cat odps_data_info.json

# 3. 检查日志文件
cat log/odps_log_check
```

## 下一步计划

### 可能的扩展

1. **缓存机制**: 缓存查询结果，避免重复查询
2. **增量加载**: 支持增量更新数据
3. **分布式训练**: 支持多 GPU 训练
4. **在线预测**: 实时预测接口
5. **特征工程**: 利用 dym_feat_feat 等历史流量特征

### 性能优化

1. **批量查询**: 一次查询多个城市
2. **并行加载**: 并行处理多个节点
3. **内存优化**: 使用生成器减少内存占用
4. **查询优化**: 优化 SQL 查询性能

## 文件清单

### 新增文件
```
lib/odps_data_loader.py          # ODPS 数据加载器
lib/data_loader.py               # 通用数据加载器
train_odps.py                    # ODPS 训练脚本
config/ODPS.conf                 # ODPS 配置文件
quickstart_odps.py               # 快速开始脚本
check_odps_data.py               # 数据检查脚本
ODPS_TRAINING_GUIDE.md           # ODPS 训练指南
DATA_LOADER_README.md            # 数据加载器文档
SUMMARY.md                       # 本文档
```

### 修改文件
```
main.py                          # 更新为使用 DataLoader
README.md                        # 添加 ODPS 使用说明
```

## 总结

本次修改成功地为 PatchSTG 项目添加了从 MaxCompute 直接加载数据的能力，使得：

✅ 可以使用生产环境的实时数据
✅ 支持动态选择城市和日期范围
✅ 保持与原有代码的完全兼容
✅ 提供完整的文档和工具支持
✅ 便于后续扩展和维护

项目现在支持两种数据源：
1. **NPZ 文件** - 适合研究和复现论文结果
2. **ODPS 表** - 适合生产环境和大规模实验

两种方式使用相同的模型架构和训练流程，可以无缝切换。
