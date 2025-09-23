"""
训练配置管理
"""

from pydantic_settings import BaseSettings
from typing import Literal, Optional, Dict, Any, Union, Type
from pathlib import Path


class TrainerConfig(BaseSettings):
    """
    训练配置类，支持从环境变量、配置文件或命令行参数加载
    """
    
    # ==================== 基础参数 ====================
    seed: int = 159456
    """随机种子，用于确保实验的可重复性。影响数据加载、模型初始化等随机过程"""
    
    epochs: int = 50
    """训练轮数，即模型遍历整个训练数据集的次数"""
    
    batch_size: int = 32
    """批次大小，每次训练使用的样本数量。较大的批次可以提高训练稳定性但需要更多内存"""
    
    accumulate_grad_batches: int = 1
    """梯度累积步数。当GPU内存不足时，可以累积多个小批次的梯度再更新参数"""
    
    num_workers: int = 4
    """数据加载器的工作进程数。0表示在主进程中加载数据，>0表示使用多进程并行加载"""
    
    lr: float = 1e-3
    """学习率，控制参数更新的步长。过大会导致训练不稳定，过小会收敛缓慢"""
    
    model_or_path: Union[str, Type] = ''
    """模型类或模型文件路径。可以是模型类（直接传入）或模型文件路径（字符串）"""
    
    dataset_or_path: Union[str, Type] = ''
    """数据集类或数据集文件路径。可以是数据集类（直接传入）或数据集文件路径（字符串）"""
    
    # ==================== 模型和数据集参数 ====================
    model_kwargs: Dict[str, Any] = {}
    """模型初始化参数字典。这些参数会传递给模型的构造函数"""
    
    data_kwargs: Dict[str, Any] = {}
    """数据集初始化参数字典。这些参数会传递给数据集的构造函数"""
    
    # ==================== 训练参数 ====================
    log_dir: str = './logs'
    """日志保存目录。训练日志、模型检查点等文件会保存在此目录下"""
    
    loss_function: str = 'bce'
    """损失函数类型。目前支持 'bce'（二元交叉熵）等"""
    
    pretrain_checkpoint_path: Optional[str] = None
    """预训练模型检查点路径。如果提供，会从此路径加载预训练权重"""
    
    # ==================== 监控器参数 ====================
    monitor_metric: str = 'val_loss'
    """监控指标名称。用于模型检查点保存和早停策略的指标"""
    
    monitor_mode: Literal['max', 'min'] = 'min'
    """监控模式。'min'表示指标越小越好（如损失），'max'表示指标越大越好（如准确率）"""
    
    use_early_stopping: bool = False
    """是否启用早停策略。当验证指标不再改善时提前停止训练"""
    
    early_stopping_patience: int = 10
    """早停耐心值。验证指标连续多少个epoch不改善就停止训练"""
    
    early_stopping_min_delta: float = 0.001
    """早停最小改善阈值。只有改善超过此阈值才认为是有效改善"""
    
    use_swa: bool = True
    """是否启用随机权重平均（Stochastic Weight Averaging）。可以提升模型泛化能力"""
    
    swa_lrs: float = 0.0005
    """SWA学习率。在SWA阶段使用的学习率"""
    
    swa_epoch_start: float = 0.9
    """SWA开始时机。在总训练轮数的多少比例时开始SWA（0.9表示90%时开始）"""
    
    # ==================== 优化器 & 调度器 ====================
    optimizer: Literal['adam', 'sgd', 'rmsprop', 'adamw'] = 'sgd'
    """优化器类型。支持 adam、sgd、rmsprop、adamw 四种优化器"""
    
    weight_decay: float = 1e-5
    """权重衰减（L2正则化）系数。用于防止过拟合"""
    
    lr_scheduler: Literal['step', 'cosine', 'None'] = 'cosine'
    """学习率调度器类型。'step'为阶梯式衰减，'cosine'为余弦退火，'None'为不调度"""
    
    lr_decay_steps: int = 20
    """学习率衰减步数。当使用'step'调度器时，每多少轮衰减一次"""
    
    lr_decay_rate: float = 0.5
    """学习率衰减率。当使用'step'调度器时，每次衰减的倍数"""
    
    lr_decay_min_lr: float = 1e-5
    """最小学习率。学习率不会低于此值"""
    
    # ==================== 设备配置 ====================
    device: str = 'auto'
    """计算设备。'auto'自动选择，'cpu'使用CPU，'cuda'使用GPU，'mps'使用Apple Silicon"""
    
    precision: str = '32'
    """数值精度。'16'为半精度（节省内存），'32'为单精度（默认），'64'为双精度"""
    
    class Config:
        """Pydantic配置类"""
        protected_namespaces = ('settings_',)
        env_prefix = 'PL_'  # 环境变量前缀，如 PL_LR=0.001
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainerConfig':
        """
        从字典创建配置实例
        
        参数:
            config_dict (Dict[str, Any]): 包含配置参数的字典
            
        返回:
            TrainerConfig: 配置实例
            
        示例:
            config_dict = {'epochs': 100, 'lr': 0.01, 'batch_size': 64}
            config = TrainerConfig.from_dict(config_dict)
        """
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TrainerConfig':
        """
        从配置文件创建配置实例
        
        参数:
            config_path (str): 配置文件路径，支持 .json 和 .yaml/.yml 格式
            
        返回:
            TrainerConfig: 配置实例
            
        示例:
            # 从JSON文件加载
            config = TrainerConfig.from_file('config.json')
            
            # 从YAML文件加载
            config = TrainerConfig.from_file('config.yaml')
        """
        config_path = Path(config_path)
        if config_path.suffix == '.json':
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        返回:
            Dict[str, Any]: 包含所有配置参数的字典
            
        示例:
            config_dict = config.to_dict()
            print(config_dict['epochs'])  # 50
        """
        return self.dict()
    
    def save_to_file(self, config_path: str) -> None:
        """
        将配置保存到文件
        
        参数:
            config_path (str): 保存路径，支持 .json 和 .yaml/.yml 格式
            
        示例:
            # 保存为JSON格式
            config.save_to_file('my_config.json')
            
            # 保存为YAML格式
            config.save_to_file('my_config.yaml')
        """
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        if config_path.suffix == '.json':
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def update(self, **kwargs) -> 'TrainerConfig':
        """
        更新配置参数并返回新的配置实例
        
        参数:
            **kwargs: 要更新的参数，如 epochs=100, lr=0.01
            
        返回:
            TrainerConfig: 更新后的新配置实例
            
        示例:
            # 更新部分参数
            new_config = config.update(epochs=100, lr=0.01)
            
            # 原配置不会被修改
            print(config.epochs)  # 仍然是 50
            print(new_config.epochs)  # 100
        """
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.__class__(**current_dict)
