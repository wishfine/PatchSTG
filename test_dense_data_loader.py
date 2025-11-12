"""
测试新的数据加载逻辑
验证数据格式是否正确（密集格式）
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# 设置环境变量（如果还没设置）
if not os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'):
    print("⚠️  请先设置 ODPS 环境变量")
    sys.exit(1)

from lib.odps_data_loader import ODPSDataLoader
from lib.utils import log_string

print("=" * 80)
print("🧪 测试新的数据加载逻辑（密集格式）")
print("=" * 80)

# 创建日志
log = open('./log/test_data_loader.log', 'w')

# 配置
config = {
    'odps_project': 'autonavi_traffic_report',
    'odps_endpoint': 'http://service-corp.odps.aliyun-inc.com/api',
    'odps_table': 'tb_inter_spatial_method_pretrain_data',
    'odps_meta_table': 'intersection_meta_aligned',
    'adcode': '650100',
    'start_date': '20250919',
    'end_date': '20250919',  # 只测试1天数据
    'input_len': 12,
    'output_len': 12,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'recur_times': 1,
    'spa_patchsize': 4
}

print("\n📋 配置:")
print(f"  城市: {config['adcode']} (乌鲁木齐)")
print(f"  日期: {config['start_date']} ~ {config['end_date']}")
print(f"  输入长度: {config['input_len']}")
print(f"  输出长度: {config['output_len']}")
print(f"  数据划分: {config['train_ratio']:.0%} / {config['val_ratio']:.0%} / {config['test_ratio']:.0%}")

# 创建数据加载器
print("\n" + "=" * 80)
print("📊 加载数据...")
print("=" * 80)

try:
    loader = ODPSDataLoader(config, log)
    loader.load_data()
    
    print("\n✅ 数据加载成功！")
    
except Exception as e:
    print(f"\n❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    log.close()
    sys.exit(1)

# 获取数据
trainX, trainY, trainXTE, trainYTE = loader.get_train_data()
valX, valY, valXTE, valYTE = loader.get_val_data()
testX, testY, testXTE, testYTE = loader.get_test_data()
mean, std = loader.get_normalization_params()

# 验证数据形状
print("\n" + "=" * 80)
print("📏 数据形状验证")
print("=" * 80)

print(f"\n训练集:")
print(f"  X shape: {trainX.shape}")
print(f"  Y shape: {trainY.shape}")
print(f"  TE shape: {trainXTE.shape}")
print(f"  期望: (samples, 12, {loader.node_num}, 1)")

print(f"\n验证集:")
print(f"  X shape: {valX.shape}")
print(f"  Y shape: {valY.shape}")

print(f"\n测试集:")
print(f"  X shape: {testX.shape}")
print(f"  Y shape: {testY.shape}")

# 验证是否是密集格式（关键测试！）
print("\n" + "=" * 80)
print("🔍 密集性验证（关键）")
print("=" * 80)

# 检查每个样本有多少节点有非零值
def check_density(X, name):
    print(f"\n{name}:")
    # 对于每个样本，统计有多少节点在任意时间步有非零值
    nodes_with_data = []
    for i in range(min(5, len(X))):  # 检查前5个样本
        # X[i]: (T, N, 1)
        has_data = (X[i, :, :, 0] > 0).any(axis=0)  # (N,) - 每个节点是否有数据
        num_nodes = has_data.sum()
        nodes_with_data.append(num_nodes)
        
        if i < 3:  # 详细显示前3个样本
            print(f"  样本 {i}: {num_nodes}/{loader.node_num} 个节点有数据 ({num_nodes/loader.node_num*100:.1f}%)")
            
            # 显示样本的流量范围
            sample_data = X[i, :, :, 0]
            non_zero = sample_data[sample_data > 0]
            if len(non_zero) > 0:
                print(f"         流量范围: [{non_zero.min():.2f}, {non_zero.max():.2f}]")
    
    avg_nodes = np.mean(nodes_with_data)
    min_nodes = np.min(nodes_with_data)
    max_nodes = np.max(nodes_with_data)
    
    print(f"  平均: {avg_nodes:.1f} 个节点有数据 ({avg_nodes/loader.node_num*100:.1f}%)")
    print(f"  范围: [{min_nodes}, {max_nodes}]")
    
    # 判断是否密集
    if avg_nodes / loader.node_num > 0.5:
        print(f"  ✅ 数据是密集的（> 50% 节点有数据）")
        return True
    elif avg_nodes / loader.node_num > 0.1:
        print(f"  ⚠️  数据稀疏度中等（10-50% 节点有数据）")
        return False
    else:
        print(f"  ❌ 数据是稀疏的（< 10% 节点有数据）")
        return False

is_dense_train = check_density(trainX, "训练集")
is_dense_val = check_density(valX, "验证集")

# 验证数据值
print("\n" + "=" * 80)
print("📊 数据统计")
print("=" * 80)

print(f"\n流量值统计:")
print(f"  训练集范围: [{trainX.min():.2f}, {trainX.max():.2f}]")
print(f"  验证集范围: [{valX.min():.2f}, {valX.max():.2f}]")
print(f"  测试集范围: [{testX.min():.2f}, {testX.max():.2f}]")

print(f"\n归一化参数:")
print(f"  Mean: {mean:.4f}")
print(f"  Std: {std:.4f}")

# 非零值比例
train_nonzero_ratio = (trainX > 0).sum() / trainX.size * 100
val_nonzero_ratio = (valX > 0).sum() / valX.size * 100

print(f"\n非零值比例:")
print(f"  训练集: {train_nonzero_ratio:.2f}%")
print(f"  验证集: {val_nonzero_ratio:.2f}%")

# 时间特征验证
print("\n" + "=" * 80)
print("⏰ 时间特征验证")
print("=" * 80)

print(f"\n前3个样本的时间特征:")
for i in range(min(3, len(trainXTE))):
    te = trainXTE[i]
    print(f"\n样本 {i}:")
    print(f"  输入时间步 0: tod={te[0,0]:.3f}, dow={te[0,1]:.3f}")
    print(f"  输入时间步 11: tod={te[11,0]:.3f}, dow={te[11,1]:.3f}")

# 空间 patch 验证
print("\n" + "=" * 80)
print("🌳 空间 Patch 验证")
print("=" * 80)

ori_parts, reo_parts, reo_all = loader.get_patch_indices()

if ori_parts is not None:
    print(f"  Patch 数量: {len(ori_parts)}")
    patch_sizes = [len(p) for p in ori_parts]
    print(f"  Patch 大小范围: [{min(patch_sizes)}, {max(patch_sizes)}]")
    print(f"  平均 Patch 大小: {np.mean(patch_sizes):.1f}")
    print(f"  ✅ 空间 patch 创建成功")
else:
    print(f"  ❌ 未创建空间 patch")

# 最终判定
print("\n" + "=" * 80)
print("✅ 测试结果")
print("=" * 80)

success = True
issues = []

# 1. 形状检查
if trainX.shape[1:] != (12, loader.node_num, 1):
    success = False
    issues.append(f"❌ 训练集形状错误: {trainX.shape}")
else:
    print("✅ 数据形状正确")

# 2. 密集性检查
if not is_dense_train:
    success = False
    issues.append("❌ 数据仍然是稀疏的（每个样本只有少数节点有值）")
else:
    print("✅ 数据是密集的（每个样本包含大部分节点）")

# 3. 数据有效性检查
if np.any(np.isnan(trainX)) or np.any(np.isinf(trainX)):
    success = False
    issues.append("❌ 数据包含 NaN 或 Inf")
else:
    print("✅ 数据无 NaN/Inf")

# 4. 样本数量合理性
expected_samples = 1440 - 12 - 12 + 1  # 1天1440分钟
actual_total = len(trainX) + len(valX) + len(testX)
if abs(actual_total - expected_samples) > 10:
    print(f"⚠️  样本数量异常: 期望约{expected_samples}，实际{actual_total}")
else:
    print(f"✅ 样本数量合理: {actual_total}")

# 5. Patch 检查
if ori_parts is None:
    success = False
    issues.append("❌ 未创建空间 patch")
else:
    print("✅ 空间 patch 已创建")

print("\n" + "=" * 80)
if success:
    print("🎉 所有测试通过！数据格式正确，可以开始训练。")
else:
    print("⚠️  存在问题:")
    for issue in issues:
        print(f"  {issue}")
print("=" * 80)

log.close()

print(f"\n详细日志已保存到: ./log/test_data_loader.log")
