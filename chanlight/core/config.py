"""
训练配置管理
"""

from pydantic_settings import BaseSettings
from typing import Literal, Optional, Dict, Any
from pathlib import Path


class TrainerConfig(BaseSettings):
    """训练配置类，支持从环境变量、配置文件或命令行参数加载"""
    
    # 基础参数
    seed: int = 159456
    epochs: int = 50
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    lr: float = 1e-3
    model_name: str = ''
    dataset_name: str = ''
    
    # 模型和数据集参数
    model_kwargs: Dict[str, Any] = {}
    data_kwargs: Dict[str, Any] = {}
    
    # 训练参数
    log_dir: str = './logs'
    loss: str = 'bce'
    pretrain_checkpoint_path: Optional[str] = None
    
    # 监控器参数
    monitor_metric: str = 'val_loss'
    monitor_mode: Literal['max', 'min'] = 'min'
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    use_swa: bool = True
    swa_lrs: float = 0.0005
    swa_epoch_start: float = 0.9
    
    # 优化器 & 调度器
    optimizer: Literal['adam', 'sgd', 'rmsprop', 'adamw'] = 'sgd'
    weight_decay: float = 1e-5
    lr_scheduler: Literal['step', 'cosine', 'None'] = 'cosine'
    lr_decay_steps: int = 20
    lr_decay_rate: float = 0.5
    lr_decay_min_lr: float = 1e-5
    
    # 设备配置
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    precision: str = '32'  # '16', '32', '64'
    
    class Config:
        protected_namespaces = ('settings_',)
        env_prefix = 'PL_'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainerConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TrainerConfig':
        """从配置文件创建配置"""
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
        """转换为字典"""
        return self.dict()
    
    def save_to_file(self, config_path: str) -> None:
        """保存配置到文件"""
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
        """更新配置参数"""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.__class__(**current_dict)
