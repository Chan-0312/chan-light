"""
核心模块 - 配置管理和训练器
"""

from chanlight.core.config import TrainerConfig
from chanlight.core.trainer import PLTrainer
from chanlight.core.model_interface import ModelInterface
from chanlight.core.data_interface import DataInterface
from chanlight.core.training_manager import TrainingManager


__all__ = [
    'TrainerConfig',
    'PLTrainer', 
    'ModelInterface',
    'DataInterface',
    'TrainingManager',
]
