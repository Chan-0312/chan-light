"""
训练管理器 - 高级训练接口
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from chanlight.core.config import TrainerConfig
from chanlight.core.trainer import PLTrainer
from chanlight.core.model_interface import setup_model_interface
from chanlight.core.data_interface import setup_data_interface


class TrainingManager:
    """训练管理器，提供高级训练接口"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.trainer = PLTrainer(config)
        self.model = None
        self.data_module = None
        
    
    def setup(self, model_or_path=None, dataset_or_path=None, 
              model_kwargs: Dict[str, Any] = None, data_kwargs: Dict[str, Any] = None,
              common_step=None):
        """设置模型和数据模块
        
        Args:
            model_or_path: 模型类（直接传入）或模型文件路径（字符串）
            dataset_or_path: 数据集类（直接传入）或数据集文件路径（字符串）
            model_kwargs: 模型参数（会覆盖配置中的model_kwargs）
            data_kwargs: 数据参数（会覆盖配置中的data_kwargs）
            common_step: 训练步骤函数（可选，当直接传入模型类时需要）
        """
        # 合并参数：方法参数 > 配置参数
        final_model_kwargs = {**self.config.model_kwargs, **(model_kwargs or {})}
        final_data_kwargs = {**self.config.data_kwargs, **(data_kwargs or {})}
        
        self.model = setup_model_interface(model_or_path, final_model_kwargs, common_step, self.config)
        self.data_module = setup_data_interface(dataset_or_path, final_data_kwargs, self.config)
        
        # 设置训练器
        self.trainer.setup(self.model, self.data_module)
    
    def train(self, ckpt_path: Optional[str] = None) -> Dict[str, Any]:
        """开始训练"""
        if self.model is None or self.data_module is None:
            raise ValueError("请先调用 setup() 方法设置模型和数据模块")
        
        return self.trainer.train(ckpt_path)
    
    def test(self, ckpt_path: Optional[str] = None) -> Dict[str, Any]:
        """测试模型"""
        if self.model is None or self.data_module is None:
            raise ValueError("请先调用 setup() 方法设置模型和数据模块")
        
        return self.trainer.test(ckpt_path)
    
    def predict(self, dataloader, ckpt_path: Optional[str] = None) -> List[Any]:
        """预测"""
        if self.model is None or self.data_module is None:
            raise ValueError("请先调用 setup() 方法设置模型和数据模块")
        
        return self.trainer.predict(dataloader, ckpt_path)
    
    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        """恢复训练"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        return self.train(checkpoint_path)
    
    def save_config(self, config_path: str) -> None:
        """保存配置"""
        self.config.save_to_file(config_path)
    
    def load_config(self, config_path: str) -> None:
        """加载配置"""
        self.config = TrainerConfig.from_file(config_path)
        self.trainer = PLTrainer(self.config)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        # 直接调用 ModelInterface 的 get_model_info 方法
        return self.model.get_model_info()
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        if self.data_module is None:
            return {}
        
        return self.data_module.get_dataset_info()
    
    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        return {
            "config": self.config.to_dict(),
            "model_info": self.get_model_info(),
            "data_info": self.get_data_info()
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'TrainingManager':
        """从配置文件创建训练管理器"""
        config = TrainerConfig.from_file(config_path)
        return cls(config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingManager':
        """从字典创建训练管理器"""
        config = TrainerConfig.from_dict(config_dict)
        return cls(config)
