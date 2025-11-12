# 日志管理指南

## ⚠️ 当前行为

**现在的情况：每次运行 `train_model.ipynb` 会覆盖之前的日志文件！**

当前代码使用写入模式 (`'w'`) 打开日志文件：
```python
log = open(args['log_file'], 'w')  # 'w' 模式会清空并覆盖文件
```

这意味着：
- ✅ 每次训练都有干净的日志
- ❌ 之前的训练记录会丢失
- ❌ 无法对比多次训练结果

---

## 💡 解决方案

### 方案 1：手动备份日志（最简单）

**每次训练前手动备份：**

```bash
# 在服务器上运行训练前
cp ./log/odps_beijing_log ./log/odps_beijing_log_backup_$(date +%Y%m%d_%H%M%S)

# 然后再训练
jupyter notebook train_model.ipynb
```

**优点：** 简单直接，不需要修改代码  
**缺点：** 容易忘记，手动操作

---

### 方案 2：修改为追加模式（推荐）

修改 `train_model.ipynb` 中的第 4 个代码单元格（"4. 初始化日志"）：

**原来的代码：**
```python
# 打开日志文件
log = open(args['log_file'], 'w')
```

**改为追加模式：**
```python
# 打开日志文件（追加模式，不覆盖）
log = open(args['log_file'], 'a')

# 添加分隔符，区分不同的训练运行
log_string(log, '\n\n' + '=' * 100)
log_string(log, f'新的训练运行开始 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log_string(log, '=' * 100 + '\n')
```

**优点：** 自动保留所有历史记录  
**缺点：** 日志文件会越来越大

---

### 方案 3：按时间戳创建新日志（最灵活）

修改配置部分，为每次训练创建独立的日志文件：

**修改第 2 个代码单元格（"2. 配置参数"）：**

```python
# ... 其他配置 ...

# 文件路径（使用时间戳）
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
args = {
    # ... 其他参数 ...
    'model_file': f'./saved_models/odps_beijing_model_{timestamp}.pth',
    'log_file': f'./log/odps_beijing_log_{timestamp}',
}
```

这样每次训练会生成：
- `./log/odps_beijing_log_20251112_153000`
- `./log/odps_beijing_log_20251112_183000`
- `./log/odps_beijing_log_20251113_093000`

**优点：**
- ✅ 完整保留所有历史记录
- ✅ 每次训练的日志独立存储
- ✅ 易于对比不同训练结果

**缺点：**
- 需要修改配置代码
- 每次训练的模型文件也会不同

---

### 方案 4：组合方案（生产级）

结合方案 2 和方案 3：
- 主日志使用追加模式（保留完整历史）
- 同时为每次训练创建独立的时间戳日志（方便查看单次训练）

```python
from datetime import datetime
import sys

# 时间戳
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 配置
args = {
    # ... 其他参数 ...
    'model_file': f'./saved_models/odps_beijing_model_best.pth',  # 最佳模型固定名称
    'log_file': './log/odps_beijing_log',  # 主日志（追加模式）
    'detailed_log_file': f'./log/odps_beijing_log_{timestamp}',  # 详细日志（本次训练）
}

# 打开两个日志文件
log_main = open(args['log_file'], 'a')  # 追加到主日志
log_detail = open(args['detailed_log_file'], 'w')  # 本次训练的详细日志

# 包装 log_string 以同时写入两个文件
class DualLogger:
    def __init__(self, main_log, detail_log):
        self.main = main_log
        self.detail = detail_log
    
    def write(self, message):
        self.main.write(message)
        self.main.flush()
        self.detail.write(message)
        self.detail.flush()
    
    def writable(self):
        return True
    
    def close(self):
        self.main.close()
        self.detail.close()

log = DualLogger(log_main, log_detail)

# 在主日志中添加分隔符
log_string(log, '\n\n' + '=' * 100)
log_string(log, f'训练运行 #{timestamp}')
log_string(log, '=' * 100 + '\n')
```

**优点：** 功能最完整，适合长期项目  
**缺点：** 代码稍复杂

---

## 📊 查看历史日志

### 如果使用方案 1（手动备份）

```bash
# 查看所有备份
ls -lh ./log/odps_beijing_log_backup_*

# 查看特定备份
cat ./log/odps_beijing_log_backup_20251112_153000
```

### 如果使用方案 2（追加模式）

```bash
# 查看完整日志
cat ./log/odps_beijing_log

# 搜索所有训练运行的开始
grep "新的训练运行开始" ./log/odps_beijing_log

# 查看最近一次训练（从最后一个分隔符开始）
sed -n '/新的训练运行开始/h;//!H;$!d;x;p' ./log/odps_beijing_log | tail -200
```

### 如果使用方案 3（时间戳日志）

```bash
# 列出所有训练日志
ls -lht ./log/odps_beijing_log_*

# 查看最新的训练日志
cat $(ls -t ./log/odps_beijing_log_* | head -1)

# 对比两次训练
diff ./log/odps_beijing_log_20251112_153000 ./log/odps_beijing_log_20251112_183000
```

---

## 🎯 推荐做法

### 对于实验阶段（现在）

**推荐：方案 3（时间戳日志）**

理由：
- ✅ 每次实验都有独立记录
- ✅ 方便对比不同参数/配置的效果
- ✅ 不会意外覆盖重要结果
- ✅ 易于清理旧日志

### 对于生产环境（未来）

**推荐：方案 4（组合方案）**

理由：
- ✅ 主日志记录所有历史（用于审计）
- ✅ 详细日志方便单独查看
- ✅ 专业且可靠

---

## 📝 快速修改步骤（方案 3）

如果你想立即使用时间戳日志（最简单推荐）：

1. **打开 `train_model.ipynb`**

2. **找到第 2 个代码单元格（"2. 配置参数"）**

3. **在配置字典定义前添加：**
   ```python
   from datetime import datetime
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   ```

4. **修改这两行：**
   ```python
   # 原来：
   'model_file': './saved_models/odps_beijing_model.pth',
   'log_file': './log/odps_beijing_log',
   
   # 改为：
   'model_file': f'./saved_models/odps_beijing_model_{timestamp}.pth',
   'log_file': f'./log/odps_beijing_log_{timestamp}',
   ```

5. **保存并运行**

完成！现在每次训练都会生成独立的日志和模型文件。

---

## 🧹 清理旧日志

如果日志文件太多：

```bash
# 只保留最近 10 个日志
cd ./log
ls -t odps_beijing_log_* | tail -n +11 | xargs rm -f

# 删除 7 天前的日志
find ./log -name "odps_beijing_log_*" -mtime +7 -delete

# 压缩旧日志（保留但节省空间）
gzip ./log/odps_beijing_log_202511*
```

---

## ❓ 常见问题

**Q: 我之前的训练日志丢了，能恢复吗？**  
A: 如果已经被覆盖，无法恢复。建议从现在开始使用方案 3。

**Q: 日志文件会很大吗？**  
A: 单次训练的日志约 10-100 KB，50 个 epoch 也就几百 KB。如果使用追加模式，定期清理旧记录即可。

**Q: 模型文件也会被覆盖吗？**  
A: 是的！如果使用固定的模型文件名，每次训练会覆盖。建议同样使用时间戳命名，或者只保存最佳模型（当前做法）。

**Q: 怎么知道哪个日志对应哪个模型？**  
A: 如果使用方案 3，日志和模型的时间戳相同。或者在日志开头记录模型文件路径。

---

## 📚 相关文档

- `train_model.ipynb` - 训练 Notebook
- `SERVER_DEPLOYMENT_GUIDE.md` - 服务器部署指南
- `JUPYTER_NOTEBOOKS_GUIDE.md` - Jupyter 使用指南
