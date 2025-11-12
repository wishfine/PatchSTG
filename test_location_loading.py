"""
测试经纬度加载
验证元数据表中的位置信息是否可用
"""
import os
import sys
from odps import ODPS

# 测试环境变量
print("=" * 80)
print("🧪 测试经纬度加载")
print("=" * 80)

access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

if not access_id or not secret:
    print("❌ 缺少 ODPS 凭证环境变量")
    sys.exit(1)

print(f"✅ Access ID: {access_id[:10]}...")
print()

# ODPS 配置
PROJECT = 'autonavi_traffic_report'
ENDPOINT = 'http://service-corp.odps.aliyun-inc.com/api'
TABLE_META = 'intersection_meta_aligned'  # 使用对齐的元数据表
ADCODE = '650100'

# 连接 ODPS
print("📡 连接 ODPS...")
odps = ODPS(access_id, secret, PROJECT, endpoint=ENDPOINT)
print(f"✅ 连接成功: {PROJECT}\n")


# 测试 1: 检查元数据表结构
print("=" * 80)
print("📋 测试 1: 检查元数据表结构")
print("=" * 80)

query_schema = f"""
DESC {TABLE_META}
"""

try:
    table = odps.get_table(TABLE_META)
    print(f"表名: {TABLE_META}")
    print(f"字段列表:")
    for col in table.table_schema.columns:
        print(f"  - {col.name}: {col.type}")
    print()
except Exception as e:
    print(f"❌ 获取表结构失败: {e}")
    sys.exit(1)


# 测试 2: 检查字段名称
print("=" * 80)
print("🔍 测试 2: 检查位置字段名称")
print("=" * 80)

query_sample = f"""
SELECT *
FROM {TABLE_META}
WHERE adcode = '{ADCODE}'
LIMIT 3
"""

print(f"查询: {query_sample}\n")

try:
    with odps.execute_sql(query_sample).open_reader() as reader:
        columns = [col.name for col in reader._schema.columns]
        print(f"表中的字段名: {columns}\n")
        
        # 检查是否有位置字段
        lat_fields = [col for col in columns if 'lat' in col.lower()]
        lng_fields = [col for col in columns if 'lng' in col.lower() or 'lon' in col.lower()]
        
        print(f"纬度相关字段: {lat_fields}")
        print(f"经度相关字段: {lng_fields}\n")
        
        if not lat_fields or not lng_fields:
            print("⚠️  警告: 没有找到明确的经纬度字段！")
            print("请手动确认字段名称。\n")
        
        # 显示前3条记录
        print("前 3 条记录:")
        for i, record in enumerate(reader):
            print(f"\n记录 {i+1}:")
            for col_name in columns:
                value = record[col_name]
                print(f"  {col_name}: {value}")
                
except Exception as e:
    print(f"❌ 查询失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# 测试 3: 统计位置覆盖率
print("\n" + "=" * 80)
print("📊 测试 3: 统计位置覆盖率")
print("=" * 80)

# 根据实际字段名调整查询
# 可能的字段名: lat/latitude, lng/longitude/lon
query_coverage = f"""
SELECT 
    COUNT(*) as total_count,
    SUM(CASE WHEN lat IS NOT NULL AND lng IS NOT NULL THEN 1 ELSE 0 END) as with_location,
    SUM(CASE WHEN lat IS NOT NULL AND lng IS NOT NULL AND lat != 0 AND lng != 0 THEN 1 ELSE 0 END) as with_valid_location
FROM {TABLE_META}
WHERE adcode = '{ADCODE}'
"""

print(f"查询: {query_coverage}\n")

try:
    with odps.execute_sql(query_coverage).open_reader() as reader:
        for record in reader:
            total = record['total_count']
            with_loc = record['with_location']
            with_valid = record['with_valid_location']
            
            coverage = with_loc / total * 100 if total > 0 else 0
            valid_coverage = with_valid / total * 100 if total > 0 else 0
            
            print(f"📊 统计结果:")
            print(f"  总记录数: {total}")
            print(f"  有位置信息: {with_loc} ({coverage:.2f}%)")
            print(f"  有效位置（非零）: {with_valid} ({valid_coverage:.2f}%)")
            print()
            
            if valid_coverage >= 50:
                print(f"✅ 位置覆盖率合格: {valid_coverage:.2f}% >= 50%")
            else:
                print(f"❌ 位置覆盖率不足: {valid_coverage:.2f}% < 50%")
                print(f"   PatchSTG 需要至少 50% 的节点有有效位置。")
                
except Exception as e:
    print(f"❌ 统计查询失败: {e}")
    print("可能是字段名不对，请根据上面的字段列表调整查询。")
    import traceback
    traceback.print_exc()


# 测试 4: 检查位置值范围
print("\n" + "=" * 80)
print("🌍 测试 4: 检查位置值范围")
print("=" * 80)

query_range = f"""
SELECT 
    MIN(lat) as min_lat,
    MAX(lat) as max_lat,
    AVG(lat) as avg_lat,
    MIN(lng) as min_lng,
    MAX(lng) as max_lng,
    AVG(lng) as avg_lng
FROM {TABLE_META}
WHERE adcode = '{ADCODE}'
  AND lat IS NOT NULL 
  AND lng IS NOT NULL
  AND lat != 0
  AND lng != 0
"""

print(f"查询: {query_range}\n")

try:
    with odps.execute_sql(query_range).open_reader() as reader:
        for record in reader:
            print(f"📍 位置范围:")
            print(f"  纬度: [{record['min_lat']:.6f}, {record['max_lat']:.6f}]")
            print(f"  经度: [{record['min_lng']:.6f}, {record['max_lng']:.6f}]")
            print(f"  平均纬度: {record['avg_lat']:.6f}")
            print(f"  平均经度: {record['avg_lng']:.6f}")
            print()
            
            # 检查是否在合理范围（中国范围）
            lat_ok = 15 <= record['avg_lat'] <= 55
            lng_ok = 70 <= record['avg_lng'] <= 140
            
            if lat_ok and lng_ok:
                print(f"✅ 位置范围合理（在中国境内）")
            else:
                print(f"⚠️  位置范围异常，可能不在预期区域")
                
except Exception as e:
    print(f"❌ 范围查询失败: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 80)
print("✅ 测试完成")
print("=" * 80)

print("\n📝 总结:")
print("1. 如果所有测试通过，说明元数据表可用")
print("2. 请记录正确的字段名（可能是 lat/lng 或 latitude/longitude）")
print("3. 确认位置覆盖率 >= 50%")
print("4. 在配置文件中设置: odps_meta_table = 'intersection_meta_aligned'")
print("5. 数据划分比例: 训练集 60%, 验证集 20%, 测试集 20%")
