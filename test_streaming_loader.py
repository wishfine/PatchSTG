"""
测试流式数据加载器
验证内存占用和性能
"""
import os
import sys
import time
import psutil
import numpy as np
from configparser import ConfigParser

# 添加 lib 到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.odps_data_loader import ODPSDataLoader
from lib.utils import log_string


def get_memory_usage():
    """获取当前进程内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_streaming_loader(config_path='config/ODPS.conf', limit=None):
    """
    测试流式数据加载器
    
    参数:
        config_path: 配置文件路径
        limit: 限制加载记录数（用于快速测试）
    """
    print("=" * 80)
    print("流式数据加载器测试")
    print("=" * 80)
    
    # 读取配置
    config_obj = ConfigParser()
    config_obj.read(config_path)
    
    config = dict(config_obj['data'])
    
    # 转换数值类型
    for key in ['input_len', 'output_len', 'recur_times', 'spa_patchsize']:
        if key in config:
            config[key] = int(config[key])
    
    for key in ['train_ratio', 'val_ratio', 'test_ratio']:
        if key in config:
            config[key] = float(config[key])
    
    # 如果指定 limit，覆盖配置
    if limit:
        config['limit'] = limit
        print(f"⚠️  测试模式：限制加载 {limit} 条记录\n")
    
    # 创建日志文件
    log_file = open('test_streaming_loader.log', 'w')
    
    # 记录初始内存
    mem_start = get_memory_usage()
    print(f"📊 初始内存占用: {mem_start:.2f} MB\n")
    log_string(log_file, f'Initial memory: {mem_start:.2f} MB')
    
    # 创建数据加载器
    print("🚀 创建数据加载器...")
    loader = ODPSDataLoader(config, log=log_file)
    
    mem_after_init = get_memory_usage()
    print(f"📊 初始化后内存: {mem_after_init:.2f} MB (+{mem_after_init - mem_start:.2f} MB)\n")
    
    # 加载数据
    print("📥 开始流式加载数据...\n")
    time_start = time.time()
    
    try:
        loader.load_data()
        time_end = time.time()
        
        mem_after_load = get_memory_usage()
        
        print("\n" + "=" * 80)
        print("✅ 数据加载成功！")
        print("=" * 80)
        
        # 性能统计
        print("\n📈 性能统计:")
        print(f"  ⏱️  加载耗时: {time_end - time_start:.2f} 秒")
        print(f"  💾 内存占用: {mem_after_load:.2f} MB")
        print(f"  📈 内存增量: {mem_after_load - mem_start:.2f} MB")
        print(f"  📊 峰值内存: {mem_after_load:.2f} MB")
        
        # 数据统计
        print("\n📊 数据统计:")
        info = loader.get_data_info()
        print(f"  🚂 训练样本: {info['train_samples']:,}")
        print(f"  🎯 验证样本: {info['val_samples']:,}")
        print(f"  🧪 测试样本: {info['test_samples']:,}")
        print(f"  🗺️  节点数量: {info['num_nodes']:,}")
        print(f"  📏 输入形状: {info['input_shape']}")
        print(f"  📐 输出形状: {info['output_shape']}")
        print(f"  📊 归一化参数: mean={info['mean']:.4f}, std={info['std']:.4f}")
        
        # 验证数据质量
        print("\n✅ 数据质量验证:")
        trainX, trainY, trainXTE, trainYTE = loader.get_train_data()
        
        # 检查是否有 NaN 或 Inf
        has_nan = np.any(np.isnan(trainX)) or np.any(np.isnan(trainY))
        has_inf = np.any(np.isinf(trainX)) or np.any(np.isinf(trainY))
        
        if not has_nan and not has_inf:
            print("  ✅ 无 NaN 或 Inf 值")
        else:
            print(f"  ❌ 数据异常: NaN={has_nan}, Inf={has_inf}")
        
        # 统计非零值比例
        nonzero_ratio = (trainX > 0).sum() / trainX.size * 100
        print(f"  📊 非零值比例: {nonzero_ratio:.2f}%")
        
        # 统计值分布
        print(f"  📈 流量值范围: [{trainX.min():.2f}, {trainX.max():.2f}]")
        print(f"  📊 平均流量: {trainX[trainX > 0].mean():.2f}")
        
        # Patch 信息
        try:
            ori_parts, reo_parts, reo_all = loader.get_patch_indices()
            print(f"\n🌳 空间分组信息:")
            print(f"  📦 Patch 数量: {len(ori_parts)}")
            patch_sizes = [len(p) for p in ori_parts]
            print(f"  📏 Patch 大小: min={min(patch_sizes)}, max={max(patch_sizes)}, avg={np.mean(patch_sizes):.1f}")
        except:
            print(f"\n⚠️  空间分组未创建（可能缺少位置信息）")
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 加载失败: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
    
    finally:
        log_file.close()
        print(f"\n📝 完整日志已保存到: test_streaming_loader.log")


def compare_memory_usage():
    """
    对比不同数据量下的内存占用
    """
    print("\n" + "=" * 80)
    print("内存占用对比测试")
    print("=" * 80)
    
    test_cases = [
        ("小数据", 1000),
        ("中数据", 10000),
        ("大数据", 100000),
    ]
    
    results = []
    
    for name, limit in test_cases:
        print(f"\n📊 测试 {name} ({limit:,} 条记录)...")
        
        mem_before = get_memory_usage()
        
        try:
            # 创建配置
            config_obj = ConfigParser()
            config_obj.read('config/ODPS.conf')
            config = dict(config_obj['data'])
            
            for key in ['input_len', 'output_len', 'recur_times', 'spa_patchsize']:
                if key in config:
                    config[key] = int(config[key])
            
            for key in ['train_ratio', 'val_ratio', 'test_ratio']:
                if key in config:
                    config[key] = float(config[key])
            
            config['limit'] = limit
            
            # 加载数据
            loader = ODPSDataLoader(config, log=None)
            loader.load_data()
            
            mem_after = get_memory_usage()
            mem_delta = mem_after - mem_before
            
            info = loader.get_data_info()
            
            results.append({
                'name': name,
                'records': limit,
                'samples': info['train_samples'] + info['val_samples'] + info['test_samples'],
                'memory': mem_delta
            })
            
            print(f"  ✅ 内存增量: {mem_delta:.2f} MB")
            print(f"  📊 生成样本: {info['train_samples'] + info['val_samples'] + info['test_samples']:,}")
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 内存占用汇总")
    print("=" * 80)
    print(f"{'测试用例':<10} {'记录数':>12} {'样本数':>12} {'内存增量(MB)':>15}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10} {r['records']:>12,} {r['samples']:>12,} {r['memory']:>15.2f}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试流式数据加载器')
    parser.add_argument('--config', type=str, default='config/ODPS.conf',
                       help='配置文件路径')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制加载记录数（用于快速测试）')
    parser.add_argument('--compare', action='store_true',
                       help='运行内存占用对比测试')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_memory_usage()
    else:
        test_streaming_loader(args.config, args.limit)
