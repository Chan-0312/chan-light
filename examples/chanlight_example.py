"""
装饰器使用示例 - 使用装饰器注册模型和数据集
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from chanlight import (
    TrainerConfig, 
    TrainingManager, 
    register_model, 
    register_dataset,
    get_registry
)


# 使用 @register_model 装饰器
@register_model("my_cnn")
class MyCNN(nn.Module):
    """使用装饰器注册的CNN模型"""
    
    def __init__(self, input_size=1024, hidden_size=256, num_classes=1, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
    
    @staticmethod
    def common_step(model, batch, log, hparams, stage='train'):
        """训练步骤函数"""
        inputs, labels = batch
        outputs = model(inputs)
        
        loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
        accuracy = ((outputs.squeeze() > 0.5).float() == labels.float()).float().mean()
        
        log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        log(f'{stage}_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss



# 注册数据集
@register_dataset("my_dataset")
class MyDataset(Dataset):
    """使用装饰器注册的数据集"""
    
    def __init__(self, data_size=1000, input_dim=1024, stage='train'):
        self.data_size = data_size
        self.input_dim = input_dim
        self.stage = stage
        
        # 生成模拟数据
        self.data = torch.randn(data_size, input_dim)
        self.labels = torch.randint(0, 2, (data_size,))
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def main():
    """装饰器使用示例"""
    
    print("=== 装饰器注册模型和数据集示例 ===\n")
    
    # 1. 查看注册的模型和数据集
    registry = get_registry()
    print("已注册的模型:", registry.list_models())
    print("已注册的数据集:", registry.list_datasets())
    print()
    
    # 2. 使用注册的模型进行训练
    print("使用注册的模型进行训练...")
    
    # 创建配置
    config = TrainerConfig(
        model_name='my_cnn',  # 使用注册的模型名
        dataset_name='my_dataset',  # 使用注册的数据集名
        epochs=3,
        batch_size=32,
        lr=0.001,
        model_kwargs={
            'input_size': 1024,
            'hidden_size': 256,
            'num_classes': 1,
            'dropout_rate': 0.3
        },
        data_kwargs={
            'data_size': 1000,
            'input_dim': 1024
        }
    )
    
    print("配置信息:")
    print(f"  模型: {config.model_name}")
    print(f"  数据集: {config.dataset_name}")
    print(f"  训练轮数: {config.epochs}")
    print()
    
    # 创建训练管理器
    manager = TrainingManager(config)
    
    # 设置模型和数据（会自动使用注册的模型和数据集）
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
    
    print("\n=== 装饰器示例完成 ===")
    print("\n装饰器优势:")
    print("1. 简洁优雅: 只需添加装饰器即可注册")
    print("2. 自动管理: 模型和数据集自动注册到全局注册器")
    print("3. 参数默认值: 可以在装饰器中设置默认参数")
    print("4. 类型安全: 装饰器会检查必需的方法")
    print("5. 易于使用: 在配置中直接使用注册的名称")


if __name__ == '__main__':
    main()
