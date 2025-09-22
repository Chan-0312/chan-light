"""
核心模块 - 配置管理和训练器
"""

from .config import TrainerConfig
from .trainer import PLTrainer
from .model_interface import ModelInterface
from .data_interface import DataInterface
from .training_manager import TrainingManager
from .decorators import (
    register_model,
    register_dataset,
    get_registry
)

__all__ = [
    'TrainerConfig',
    'PLTrainer', 
    'ModelInterface',
    'DataInterface',
    'TrainingManager',
    'register_model',
    'register_dataset',
    'get_registry'
]
