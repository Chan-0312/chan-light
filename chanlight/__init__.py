"""
ChanLight - 基于PyTorch Lightning的机器学习训练SDK
"""

from chanlight.core.config import TrainerConfig
from chanlight.core.trainer import PLTrainer
from chanlight.core.training_manager import TrainingManager
from chanlight.core.decorators import (
    register_model, 
    register_dataset, 
    get_registry
)

__version__ = "1.0.0"
__author__ = "Chan"

__all__ = [
    "TrainerConfig",
    "PLTrainer",
    "TrainingManager",
    "register_model",
    "register_dataset",
    "get_registry"
]
