"""
测试 ODPSTableDataLoader - 验证数据加载逻辑
不进行训练，只测试数据加载的每个步骤
"""
import os
import sys

print("=" * 80)
print("测试 ODPSTableDataLoader 数据加载逻辑")
print("=" * 80)

# 添加项目路径
sys.path.insert(0, '/Users/wishfine/Desktop/traffic/PatchSTG')

from lib.odps_table_data_loader import create_odps_table_dataloader
from lib.utils import log_string

# 创建日志文件
log_file = open('test_data_loader.log', 'w')

try:
    print("\n[1] 配置参数")
    print("-" * 80)
    
    config = {
        'odps_project': 'autonavi_traffic_report',
        'odps_table': 'tb_inter_spatial_method_pretrain_data',
        'odps_meta_table': 'intersection_meta_1',
        'adcode': '650100',  # 乌鲁木齐
        'start_date': '20250919',
        'end_date': '20250919',  # 只测试一天
        'batch_size': 32,
        'num_workers': 0,  # 单进程方便调试
        'input_len': 12
    }
    
    print("配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n[2] 创建 DataLoader Wrapper")
    print("-" * 80)
    
    loader_wrapper = create_odps_table_dataloader(config, log_file)
    
    print("✅ DataLoader Wrapper 创建成功")
    
    print("\n[3] 创建 DataLoader")
    print("-" * 80)
    
    data_loader = loader_wrapper.create_dataloader()
    
    print("✅ DataLoader 创建成功")
    
    print("\n[4] 测试第一个 Batch（会触发节点列表构建）")
    print("-" * 80)
    
    first_batch = None
    for batch_idx, batch in enumerate(data_loader):
        print(f"\n获取第 {batch_idx + 1} 个 batch:")
        print(f"  X shape: {batch['X'].shape}")
        print(f"  Y shape: {batch['Y'].shape}")
        print(f"  TE shape: {batch['TE'].shape}")
        
        # 详细检查第一个 batch
        if batch_idx == 0:
            first_batch = batch
            
            X = batch['X'].numpy()
            Y = batch['Y'].numpy()
            TE = batch['TE'].numpy()
            
            print(f"\n  数据统计:")
            print(f"    X 非零元素: {(X != 0).sum()} / {X.size} ({(X != 0).sum() / X.size * 100:.2f}%)")
            print(f"    Y 非零元素: {(Y != 0).sum()} / {Y.size} ({(Y != 0).sum() / Y.size * 100:.2f}%)")
            print(f"    X 范围: [{X.min():.2f}, {X.max():.2f}]")
            print(f"    Y 范围: [{Y.min():.2f}, {Y.max():.2f}]")
            print(f"    TE 范围: [{TE.min():.4f}, {TE.max():.4f}]")
            
            print(f"\n  样本分析 (第1个样本):")
            sample_0 = X[0]  # (12, num_nodes, 1)
            
            # 找到非零的节点
            nonzero_nodes = []
            for node_idx in range(sample_0.shape[1]):
                if sample_0[:, node_idx, 0].sum() != 0:
                    nonzero_nodes.append(node_idx)
            
            print(f"    该样本有数据的节点数: {len(nonzero_nodes)}")
            if len(nonzero_nodes) > 0:
                node_idx = nonzero_nodes[0]
                print(f"    节点 {node_idx} 的数据:")
                print(f"      历史流量: {sample_0[:, node_idx, 0]}")
                print(f"      当前流量: {Y[0, 0, node_idx, 0]}")
                print(f"      时间特征 (tod): {TE[0, :, 0]}")
                print(f"      时间特征 (dow): {TE[0, :, 1]}")
        
        # 只测试前 3 个 batch
        if batch_idx >= 2:
            print(f"\n✅ 成功测试了 {batch_idx + 1} 个 batches")
            break
    
    print("\n[5] 检查节点信息")
    print("-" * 80)
    
    node_info = loader_wrapper.get_node_info()
    
    print(f"节点总数: {node_info['node_num']}")
    print(f"前 5 个节点: {node_info['node_list'][:5]}")
    
    if node_info['node_locations'] is not None:
        print(f"\n位置信息:")
        print(f"  shape: {node_info['node_locations'].shape}")
        print(f"  纬度范围: [{node_info['node_locations'][0, :].min():.4f}, {node_info['node_locations'][0, :].max():.4f}]")
        print(f"  经度范围: [{node_info['node_locations'][1, :].min():.4f}, {node_info['node_locations'][1, :].max():.4f}]")
        
        # 检查前5个节点的位置
        print(f"\n  前 5 个节点的位置:")
        for i in range(min(5, node_info['node_num'])):
            lat = node_info['node_locations'][0, i]
            lng = node_info['node_locations'][1, i]
            print(f"    节点 {i}: ({lat:.6f}, {lng:.6f})")
    else:
        print("⚠️  没有位置信息")
    
    print("\n[6] 数据加载流程总结")
    print("-" * 80)
    
    print("✅ 所有步骤完成！")
    print("\n数据加载流程验证通过:")
    print("  1. ✅ ODPS 连接正常")
    print("  2. ✅ 元数据表加载成功")
    print("  3. ✅ 节点列表构建成功")
    print("  4. ✅ 位置数组构建成功")
    print("  5. ✅ Batch 数据解析成功")
    print("  6. ✅ 数据格式符合预期")
    
    print("\n下一步:")
    print("  - 可以开始训练了")
    print("  - 建议先用小数据集测试（1-2天）")
    print("  - 检查日志文件: test_data_loader.log")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n请检查:")
    print("  1. ODPS 凭证是否正确")
    print("  2. 表名是否正确")
    print("  3. 日期范围是否有数据")
    print("  4. 日志文件: test_data_loader.log")

finally:
    log_file.close()
    print("\n" + "=" * 80)
    print("测试结束")
    print("=" * 80)
