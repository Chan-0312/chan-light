# ChanLight

基于 PyTorch Lightning 的机器学习训练 SDK，使用装饰器系统简化模型和数据集注册。

## 特性

- 🎯 **装饰器注册**: 使用 `@register_model` 和 `@register_dataset` 优雅地注册模型和数据集
- ⚙️ **灵活配置**: 支持 JSON、YAML 和命令行参数配置
- 🚀 **简化训练**: 几行代码即可开始训练
- 📊 **丰富监控**: 内置 TensorBoard 日志和早停机制
- 🔧 **易于扩展**: 支持自定义模型和数据集

## 快速开始

### 1. 安装

```bash
pip install -e .
```

### 2. 定义模型和数据集

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from chanlight import register_model, register_dataset, TrainerConfig, TrainingManager

# 注册模型
@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, input_size=1024, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    @staticmethod
    def common_step(model, batch, log, hparams, mode='train'):
        inputs, labels = batch
        outputs = model(inputs)
        loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
        log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

# 注册数据集
@register_dataset("my_dataset")
class MyDataset(Dataset):
    def __init__(self, size=1000, input_dim=1024):
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### 3. 开始训练

```python
# 创建配置
config = TrainerConfig(
    model_name='my_model',      # 使用注册的模型名
    dataset_name='my_dataset',  # 使用注册的数据集名
    epochs=10,
    batch_size=32,
    lr=0.001,
    # 模型参数
    model_kwargs={
        'input_size': 1024,
        'output_size': 1
    },
    # 数据集参数
    data_kwargs={
        'size': 1000,
        'input_dim': 1024
    }
)

# 创建训练管理器
manager = TrainingManager(config)
manager.setup()

# 开始训练
results = manager.train()
print(f"训练完成! 最佳模型: {results['best_model_path']}")
```

## 装饰器系统

### 模型注册

```python
# 手动指定名称
@register_model("model_name")
class MyModel(nn.Module):
    # 模型定义
    pass

# 手动指定名称
@register_model("my_model")
class SimpleModel(nn.Module):
    # 模型定义
    pass
```

### 数据集注册

```python
# 手动指定名称
@register_dataset("dataset_name")
class MyDataset(Dataset):
    # 数据集定义
    pass

# 手动指定名称
@register_dataset("my_dataset")
class SimpleDataset(Dataset):
    # 数据集定义
    pass
```

## 配置选项

```python
config = TrainerConfig(
    # 基础参数
    epochs=50,
    batch_size=32,
    lr=0.001,
    
    # 优化器
    optimizer='adam',
    weight_decay=1e-5,
    lr_scheduler='cosine',
    
    # 监控
    use_early_stopping=True,
    early_stopping_patience=10,
    use_swa=True,
    
    # 设备
    device='auto',
    precision='32'
)
```

## 示例

查看 `examples/` 目录中的完整示例：

- `decorator_example.py` - 完整装饰器使用示例
- `simple_decorator_example.py` - 简单使用示例
- `decorator_guide.md` - 详细使用指南

## 目录结构

```
chanlight/
├── __init__.py
├── decorators.py          # 装饰器系统
├── core/                  # 核心模块
│   ├── config.py         # 配置管理
│   └── trainer.py        # 训练器封装
├── models/               # 模型管理
│   └── interface.py      # 模型接口 (ModelInterface)
├── data/                 # 数据管理
│   └── interface.py      # 数据接口
├── training/             # 训练管理
│   └── manager.py        # 训练管理器
└── utils/                # 工具模块
    ├── logger.py         # 日志工具
    └── helpers.py        # 辅助函数

examples/                 # 示例代码
├── decorator_example.py
├── simple_decorator_example.py
└── decorator_guide.md
```

## 许可证

MIT License