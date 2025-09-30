"""
ChanLight 使用示例 - 展示两种使用方式：文件路径和直接导入类
"""
import sys
sys.path.append('./')
from chanlight import TrainerConfig, TrainingManager



def test_file_path_way():
    """测试文件路径方式"""
    print("=== 方式1: 使用文件路径加载模型和数据集 ===\n")
    
    # 使用文件路径方式加载模型和数据集
    print("使用文件路径方式加载模型和数据集...")
    
    # 创建配置
    config = TrainerConfig(
        model_or_path='examples/my_model',  # 使用文件路径
        dataset_or_path='examples/my_dataset',  # 使用文件路径
        epochs=2,
        batch_size=32,
        lr=0.001,
        model_kwargs={
            'input_size': 1024,
            'hidden_size': 256,
            'num_classes': 1,
            'dropout_rate': 0.3
        },
        data_kwargs={
            'data_size': 500,
            'input_dim': 1024
        }
    )
    
    print("配置信息:")
    print(f"  模型: {config.model_or_path}")
    print(f"  数据集: {config.dataset_or_path}")
    print(f"  训练轮数: {config.epochs}")
    print()
    
    # 创建训练管理器
    manager = TrainingManager(config)
    
    # 设置模型和数据（会自动从文件加载模型和数据集）
    manager.setup()
    
    print("✓ 模型和数据设置完成")
    
    # 显示模型信息
    model_info = manager.get_model_info()
    data_info = manager.get_data_info()
    
    print(f"模型信息: {model_info}")
    print(f"数据信息: {data_info}")
    print()
    
    # 开始训练
    print("开始训练...")
    try:
        results = manager.train()
        print(f"✓ 训练完成! 最佳模型: {results.get('best_model_path', 'N/A')}")
    except Exception as e:
        print(f"! 训练演示失败 (缺少真实数据): {str(e)}")
    
    print("\n" + "="*50)


def test_direct_import_way():
    """测试直接导入类的方式"""
    print("=== 方式2: 直接导入模型和数据集类 ===\n")
    
    # 直接导入模型和数据集类
    print("直接导入模型和数据集类...")


    from my_model import MyModel
    from my_dataset import MyDataset
    
    # 创建配置
    config = TrainerConfig(
        model_or_path=MyModel,  # 直接传入模型类
        dataset_or_path=MyDataset,  # 直接传入数据集类
        epochs=2,
        batch_size=32,
        lr=0.001,
        model_kwargs={
            'input_size': 1024,
            'hidden_size': 256,
            'num_classes': 1,
            'dropout_rate': 0.3
        },
        data_kwargs={
            'data_size': 500,
            'input_dim': 1024
        }
    )
    
    print("配置信息:")
    print(f"  模型: {config.model_or_path}")
    print(f"  数据集: {config.dataset_or_path}")
    print(f"  训练轮数: {config.epochs}")
    print()
    
    # 创建训练管理器
    manager = TrainingManager(config)
    
    # 设置模型和数据（直接使用传入的类）
    manager.setup()
    
    print("✓ 模型和数据设置完成")
    
    # 显示模型信息
    model_info = manager.get_model_info()
    data_info = manager.get_data_info()
    
    print(f"模型信息: {model_info}")
    print(f"数据信息: {data_info}")
    print()
    
    # 开始训练
    print("开始训练...")
    try:
        results = manager.train()
        print(f"✓ 训练完成! 最佳模型: {results.get('best_model_path', 'N/A')}")
    except Exception as e:
        print(f"! 训练演示失败 (缺少真实数据): {str(e)}")
    
    print("\n" + "="*50)


def main():
    """主函数 - 展示两种使用方式"""
    
    # 测试方式1：文件路径
    test_file_path_way()
    
    # 测试方式2：直接导入类
    test_direct_import_way()
    
    print("\n=== 两种方式对比总结 ===")
    print("\n文件路径方式优势:")
    print("1. 简单直观: 直接指定文件路径即可")
    print("2. 自动推导: 基于文件名自动推导类名")
    print("3. 灵活扩展: 支持 .py 扩展名可选")
    print("4. 模块化: 模型和数据集可以独立管理")
    print("5. 易于维护: 代码结构清晰，便于维护")
    
    print("\n直接导入类方式优势:")
    print("1. 类型安全: IDE 可以提供更好的代码提示")
    print("2. 灵活配置: 可以在运行时动态创建类")
    print("3. 调试友好: 可以直接调试模型类")
    print("4. 性能更好: 无需动态导入文件")
    print("5. 依赖明确: 所有依赖在 import 时就能发现")


if __name__ == '__main__':
    main()
