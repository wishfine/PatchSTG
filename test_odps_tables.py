"""
测试脚本：验证能否读取 ODPS 表数据
包括主表和元数据表的关联
"""
import os
import sys

# 测试环境变量
print("=" * 60)
print("1. 检查 ODPS 凭证")
print("=" * 60)

access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

if access_id and secret:
    print(f"✅ Access ID: {access_id[:10]}...")
    print(f"✅ Secret: {'*' * 20}")
else:
    print("❌ 缺少 ODPS 凭证环境变量")
    print("请运行:")
    print("  export ALIBABA_CLOUD_ACCESS_KEY_ID='your_id'")
    print("  export ALIBABA_CLOUD_ACCESS_KEY_SECRET='your_secret'")
    sys.exit(1)


from odps import ODPS

# ODPS 配置
PROJECT = 'autonavi_traffic_report'
ENDPOINT = 'http://service-corp.odps.aliyun-inc.com/api'
TABLE_FLOW = 'tb_inter_spatial_method_pretrain_data'
TABLE_META = 'intersection_meta_1'

print("\n" + "=" * 60)
print("2. 连接 ODPS")
print("=" * 60)

try:
    odps = ODPS(access_id, secret, PROJECT, endpoint=ENDPOINT)
    print(f"✅ 连接成功: {PROJECT}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    sys.exit(1)


print("\n" + "=" * 60)
print("3. 测试读取主表（车流数据）")
print("=" * 60)

query_flow = f"""
SELECT 
    nds_id,
    next_nds_id,
    adcode,
    ds,
    passts_time,
    flow_label,
    time_feat,
    dym_feat_feat
FROM {TABLE_FLOW}
WHERE adcode = '650100'
  AND ds = '20250919'
LIMIT 5
"""

print(f"查询: {query_flow}\n")

try:
    with odps.execute_sql(query_flow).open_reader() as reader:
        print("字段: nds_id | next_nds_id | adcode | ds | passts_time | flow_label | time_feat | dym_feat_feat\n")
        
        flow_records = []
        for i, record in enumerate(reader):
            flow_records.append(record.values)
            
            nds_id = record[0]
            next_nds_id = record[1]
            adcode = record[2]
            flow_label = record[5]
            time_feat = record[6]
            dym_feat = record[7]
            
            print(f"记录 {i+1}:")
            print(f"  转向流: ({nds_id}, {next_nds_id})")
            print(f"  城市: {adcode}")
            print(f"  流量: {flow_label}")
            print(f"  time_feat 长度: {len(time_feat.split(';'))} 段")
            print(f"  dym_feat 长度: {len(dym_feat.split(';'))} 段")
            
            # 解析第一段时间特征
            first_time = time_feat.split(';')[0].split(' ')
            print(f"  首个时间特征: week={first_time[0]}, hour={first_time[1]}, minute={first_time[2]}")
            
            # 解析前5个流量值
            first_flows = dym_feat.split(';')[:5]
            print(f"  前5个历史流量: {first_flows}")
            print()
        
        print(f"✅ 主表读取成功，共 {len(flow_records)} 条记录\n")

except Exception as e:
    print(f"❌ 主表读取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


print("\n" + "=" * 60)
print("4. 测试读取元数据表（路口位置）")
print("=" * 60)

query_meta = f"""
SELECT 
    inter_id,
    nds_id,
    next_nds_id,
    lat,
    lng,
    adcode
FROM {TABLE_META}
WHERE adcode = '650100'
LIMIT 10
"""

print(f"查询: {query_meta}\n")

try:
    with odps.execute_sql(query_meta).open_reader() as reader:
        print("字段: inter_id | nds_id | next_nds_id | lat | lng | adcode\n")
        
        meta_records = []
        for i, record in enumerate(reader):
            meta_records.append(record.values)
            
            inter_id = record[0]
            nds_id = record[1]
            next_nds_id = record[2]
            lat = record[3]
            lng = record[4]
            
            print(f"记录 {i+1}:")
            print(f"  路口ID: {inter_id}")
            print(f"  转向流: ({nds_id}, {next_nds_id})")
            print(f"  位置: ({lat:.6f}, {lng:.6f})")
            print()
        
        print(f"✅ 元数据表读取成功，共 {len(meta_records)} 条记录\n")

except Exception as e:
    print(f"❌ 元数据表读取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


print("\n" + "=" * 60)
print("5. 测试关联查询（车流 + 位置）")
print("=" * 60)

query_join = f"""
SELECT 
    f.nds_id,
    f.next_nds_id,
    f.flow_label,
    m.inter_id,
    m.lat,
    m.lng
FROM {TABLE_FLOW} f
LEFT JOIN {TABLE_META} m
    ON f.nds_id = m.nds_id 
    AND f.next_nds_id = m.next_nds_id
    AND CAST(f.adcode AS STRING) = m.adcode
WHERE f.adcode = '650100'
  AND f.ds = '20250919'
LIMIT 10
"""

print(f"查询: {query_join}\n")

try:
    with odps.execute_sql(query_join).open_reader() as reader:
        print("字段: nds_id | next_nds_id | flow_label | inter_id | lat | lng\n")
        
        join_records = []
        has_location_count = 0
        
        for i, record in enumerate(reader):
            join_records.append(record.values)
            
            nds_id = record[0]
            next_nds_id = record[1]
            flow_label = record[2]
            inter_id = record[3]
            lat = record[4]
            lng = record[5]
            
            has_location = lat is not None and lng is not None
            if has_location:
                has_location_count += 1
            
            print(f"记录 {i+1}:")
            print(f"  转向流: ({nds_id}, {next_nds_id})")
            print(f"  流量: {flow_label}")
            
            if has_location:
                print(f"  路口: {inter_id}")
                print(f"  位置: ({lat:.6f}, {lng:.6f}) ✅")
            else:
                print(f"  位置: 无 ⚠️")
            print()
        
        coverage = has_location_count / len(join_records) * 100 if join_records else 0
        print(f"✅ 关联查询成功")
        print(f"📊 位置覆盖率: {coverage:.1f}% ({has_location_count}/{len(join_records)})\n")

except Exception as e:
    print(f"❌ 关联查询失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


print("\n" + "=" * 60)
print("6. 数据统计")
print("=" * 60)

# 统计主表中的转向流数量
query_stat = f"""
SELECT 
    COUNT(DISTINCT CONCAT(CAST(nds_id AS STRING), '_', CAST(next_nds_id AS STRING))) as turn_flow_count,
    COUNT(*) as total_records
FROM {TABLE_FLOW}
WHERE adcode = '650100'
  AND ds >= '20250919'
  AND ds <= '20250925'
"""

print(f"查询: {query_stat}\n")

try:
    with odps.execute_sql(query_stat).open_reader() as reader:
        for record in reader:
            turn_flow_count = record[0]
            total_records = record[1]
            
            print(f"📊 数据量统计（adcode=650100, ds=20250919~20250925）:")
            print(f"  唯一转向流数: {turn_flow_count}")
            print(f"  总记录数: {total_records}")
            print(f"  平均每个转向流的记录数: {total_records / turn_flow_count if turn_flow_count > 0 else 0:.1f}")
            print()

except Exception as e:
    print(f"⚠️  统计查询失败: {e}")


print("=" * 60)
print("✅ 所有测试完成！")
print("=" * 60)

print("\n下一步:")
print("1. 数据可以正常读取 ✅")
print("2. 元数据表有经纬度信息 ✅")
print("3. 可以使用 lib/odps_table_data_loader.py 进行训练")
print("\n使用示例:")
print("""
from lib.odps_table_data_loader import create_odps_table_dataloader

config = {
    'odps_project': 'autonavi_traffic_report',
    'odps_table': 'tb_inter_spatial_method_pretrain_data',
    'odps_meta_table': 'intersection_meta_1',
    'adcode': '650100',
    'start_date': '20250919',
    'end_date': '20250925',
    'batch_size': 64,
    'num_workers': 4,
    'input_len': 12
}

loader_wrapper = create_odps_table_dataloader(config)
data_loader = loader_wrapper.create_dataloader()

for batch in data_loader:
    X = batch['X']    # (batch, 12, nodes, 1) - 输入流量
    Y = batch['Y']    # (batch, 1, nodes, 1) - 当前流量
    TE = batch['TE']  # (batch, 12, 2) - 时间特征
    # 训练模型...
""")
