"""
ChanLight - 基于PyTorch Lightning的机器学习训练SDK
"""

from chanlight.core.config import TrainerConfig
from chanlight.core.training_manager import TrainingManager
from chanlight.core.model_interface import setup_model_interface
from chanlight.core.data_interface import setup_data_interface


__version__ = "1.0.0"
__author__ = "Chan"

__all__ = [
    "TrainerConfig",
    "TrainingManager",
    "setup_model_interface",
    "setup_data_interface",
]
