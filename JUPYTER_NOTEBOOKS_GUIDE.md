# Jupyter Notebooks 使用指南

为了方便在服务器上运行测试和检查，我们已将主要的测试脚本转换为 Jupyter Notebook 格式。

## 📓 可用的 Notebooks

### 1. `quickstart_odps.ipynb` - 快速环境检查

**用途：** 快速检查环境是否配置正确

**检查内容：**
- ✅ ODPS 凭证是否设置
- ✅ Python 依赖包是否安装
- ✅ 必要的目录是否存在

**使用方法：**
```bash
# 启动 Jupyter
jupyter notebook

# 或使用 JupyterLab
jupyter lab

# 打开 quickstart_odps.ipynb 并按顺序运行所有单元格
```

**注意事项：**
- 如果凭证未设置，可以在 Notebook 中临时设置
- ⚠️ 设置凭证后记得清除单元格输出，避免泄露

---

### 2. `test_location_loading.ipynb` - 验证元数据表

**用途：** 验证元数据表中的位置信息是否可用

**测试内容：**
1. 检查表结构和字段名
2. 查看示例数据
3. 统计位置覆盖率（需要 >= 50%）
4. 验证位置范围的合理性

**预期结果：**
```
✅ 元数据表 intersection_meta_aligned 可用
✅ 表中包含 lat 和 lng 字段
✅ 位置覆盖率: 100.00% >= 50%
✅ 位置范围合理（在中国境内）
```

**配置建议：**
- 城市代码：`650100`（乌鲁木齐）
- 元数据表：`intersection_meta_aligned`
- 节点数量：15,946 个转向流

---

### 3. `test_dense_data_loader.ipynb` - 验证数据格式

**用途：** 验证新的数据加载逻辑是否生成了密集格式的数据

**测试内容：**
1. 数据形状验证：`(samples, 12, num_nodes, 1)`
2. **密集性验证**：每个样本 > 50% 节点有数据
3. 数据有效性：无 NaN/Inf 值
4. 空间 Patch 验证
5. 样本数量合理性

**关键指标：**
```python
# 密集性检查（核心）
样本 0: 12500/15946 个节点有数据 (78.4%)  # ✅ 密集
样本 1: 12300/15946 个节点有数据 (77.1%)  # ✅ 密集

# 而不是：
样本 0: 1/15946 个节点有数据 (0.006%)     # ❌ 稀疏
```

**预期结果：**
```
✅ 数据形状正确
✅ 数据是密集的（每个样本包含大部分节点）
✅ 数据无 NaN/Inf
✅ 样本数量合理
✅ 空间 patch 已创建
🎉 所有测试通过！数据格式正确，可以开始训练。
```

---

## 🚀 使用流程

### 第一次使用（环境检查）

1. **启动 Jupyter**
   ```bash
   # 在服务器上
   jupyter notebook --ip=0.0.0.0 --port=8888
   
   # 或使用 JupyterLab
   jupyter lab --ip=0.0.0.0 --port=8888
   ```

2. **运行 `quickstart_odps.ipynb`**
   - 检查环境是否完整
   - 设置 ODPS 凭证（如果还没设置）
   - 安装缺失的包

3. **运行 `test_location_loading.ipynb`**
   - 验证元数据表可用
   - 确认位置覆盖率 >= 50%
   - 记录节点数量

4. **运行 `test_dense_data_loader.ipynb`**
   - 验证数据格式正确
   - 确认数据是密集的
   - 如果测试通过，可以开始训练

---

## 💡 使用技巧

### 1. 设置 ODPS 凭证

**方法 A：在服务器环境变量中设置（推荐）**
```bash
# 编辑 ~/.bashrc 或 ~/.zshrc
export ALIBABA_CLOUD_ACCESS_KEY_ID='your_access_key_id'
export ALIBABA_CLOUD_ACCESS_KEY_SECRET='your_access_key_secret'

# 重新加载
source ~/.bashrc
```

**方法 B：在 Notebook 中临时设置**
```python
import os
os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = 'your_id'
os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = 'your_secret'
```
⚠️ **注意：** 方法 B 需要在运行后清除单元格输出！

### 2. 安装缺失的包

在 Notebook 中直接安装：
```python
# 在 Notebook 单元格中
!pip install pyodps pandas numpy torch tqdm scikit-learn
```

### 3. 查看详细日志

测试 Notebook 会生成日志文件：
```python
# 在 Notebook 中查看
!cat ./log/test_data_loader.log

# 或者在单元格中
with open('./log/test_data_loader.log', 'r') as f:
    print(f.read())
```

### 4. 调试数据问题

如果数据格式有问题，可以在 Notebook 中交互式调试：
```python
# 检查特定样本
sample_0 = trainX[0]  # (12, num_nodes, 1)
print(f"Shape: {sample_0.shape}")

# 查看有数据的节点
has_data = (sample_0 > 0).any(axis=0).squeeze()
print(f"有数据的节点数: {has_data.sum()}/{len(has_data)}")

# 可视化（需要 matplotlib）
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.imshow(sample_0[:, :, 0].T, aspect='auto', cmap='viridis')
plt.xlabel('Time Steps')
plt.ylabel('Nodes')
plt.title('Sample 0 Traffic Flow')
plt.colorbar(label='Flow Value')
plt.show()
```

---

## 🔧 常见问题

### Q1: 如何在没有 GUI 的服务器上使用 Jupyter？

**A:** 使用端口转发：
```bash
# 在服务器上启动 Jupyter
jupyter notebook --no-browser --port=8888

# 在本地电脑上建立 SSH 隧道
ssh -L 8888:localhost:8888 username@server_ip

# 然后在本地浏览器打开 http://localhost:8888
```

### Q2: Notebook 运行很慢或卡住怎么办？

**A:** 
1. 减少测试数据量：将 `end_date` 改为与 `start_date` 相同（只测试1天）
2. 重启 Kernel：`Kernel -> Restart`
3. 清除所有输出：`Cell -> All Output -> Clear`
4. 检查内存使用：`!free -h`

### Q3: 如何保存 Notebook 的执行结果？

**A:**
```bash
# 导出为 HTML（包含输出）
jupyter nbconvert --to html --execute test_dense_data_loader.ipynb

# 导出为 PDF（需要安装 pandoc 和 LaTeX）
jupyter nbconvert --to pdf --execute test_dense_data_loader.ipynb
```

### Q4: 如何在不同环境中运行？

**A:** 使用不同的 Kernel
```bash
# 创建虚拟环境
conda create -n patchstg python=3.8
conda activate patchstg

# 安装依赖
pip install ipykernel pyodps pandas numpy torch tqdm scikit-learn

# 添加 Kernel
python -m ipykernel install --user --name=patchstg

# 在 Jupyter 中选择 patchstg Kernel
```

---

## 📊 测试数据说明

**测试配置（默认）：**
- 城市：乌鲁木齐（adcode = 650100）
- 日期：2025-09-19（1天数据用于快速测试）
- 节点数：15,946 个转向流
- 输入长度：12 个时间步
- 输出长度：12 个时间步
- 数据划分：60% / 20% / 20%

**完整训练配置：**
- 日期：2025-09-19 ~ 2025-09-25（7天数据）
- 预期样本数：约 10,000+
- 训练时间：取决于硬件和批次大小

---

## 🎯 下一步

测试通过后，可以开始训练：

```python
# 在 Python 脚本中训练
!python train_odps.py --config config/ODPS.conf --mode train

# 或在 Notebook 中训练（创建新的 training notebook）
```

如需创建训练 Notebook，可以参考 `train_odps.py` 的结构创建一个新的 `train_odps.ipynb`。

---

## 📚 相关文档

- `DATA_PIPELINE_FLOW.md` - 完整的数据流程说明
- `ODPS_TRAINING_GUIDE.md` - ODPS 训练指南
- `DATA_FORMAT_ISSUE.md` - 数据格式问题说明

---

## 🆘 需要帮助？

如果遇到问题：
1. 检查日志文件：`./log/test_data_loader.log`
2. 查看详细文档：上面列出的 Markdown 文件
3. 在 Notebook 中交互式调试：逐步运行单元格，检查中间结果
