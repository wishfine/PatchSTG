#!/usr/bin/env python
"""
快速开始脚本 - 使用 ODPS 数据训练 PatchSTG
"""
import os
import sys

def check_credentials():
    """检查 ODPS 凭证"""
    access_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    
    if not access_id or not secret:
        print("❌ 错误: 缺少 ODPS 凭证")
        print("\n请设置环境变量:")
        print("  export ALIBABA_CLOUD_ACCESS_KEY_ID='your_access_key_id'")
        print("  export ALIBABA_CLOUD_ACCESS_KEY_SECRET='your_access_key_secret'")
        print("\n或将它们添加到 ~/.zshrc 或 ~/.bashrc 中，然后运行:")
        print("  source ~/.zshrc")
        return False
    
    print("✓ ODPS 凭证已配置")
    return True

def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'odps': 'pyodps',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'torch': 'torch',
        'tqdm': 'tqdm',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n❌ 缺少依赖包，请运行:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    dirs = ['saved_models', 'log']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"✓ 创建目录: {d}")
        else:
            print(f"✓ 目录已存在: {d}")

def main():
    print("=" * 60)
    print("PatchSTG ODPS 训练 - 快速开始")
    print("=" * 60)
    print()
    
    print("步骤 1: 检查 ODPS 凭证")
    print("-" * 60)
    if not check_credentials():
        sys.exit(1)
    print()
    
    print("步骤 2: 检查依赖包")
    print("-" * 60)
    if not check_dependencies():
        sys.exit(1)
    print()
    
    print("步骤 3: 创建必要目录")
    print("-" * 60)
    create_directories()
    print()
    
    print("=" * 60)
    print("✓ 环境检查完成！")
    print("=" * 60)
    print()
    
    print("下一步操作:")
    print()
    print("1. 编辑配置文件 config/ODPS.conf")
    print("   - 设置 adcode (城市代码)")
    print("   - 设置 start_date 和 end_date (日期范围)")
    print()
    print("2. 运行训练:")
    print("   python train_odps.py --config config/ODPS.conf --mode train")
    print()
    print("3. 或运行测试:")
    print("   python train_odps.py --config config/ODPS.conf --mode test")
    print()
    print("城市代码参考:")
    print("  - 北京: 110000")
    print("  - 上海: 310000")
    print("  - 广州: 440100")
    print("  - 深圳: 440300")
    print()
    print("详细文档请查看: ODPS_TRAINING_GUIDE.md")
    print()

if __name__ == '__main__':
    main()
