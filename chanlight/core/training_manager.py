"""
训练管理器 - 高级训练接口
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from chanlight.core.config import TrainerConfig
from chanlight.core.trainer import PLTrainer
from chanlight.core.model_interface import ModelInterface
from chanlight.core.data_interface import DataInterface
from chanlight.core.decorators import registry as decorator_registry


class TrainingManager:
    """训练管理器，提供高级训练接口"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.trainer = PLTrainer(config)
        self.model = None
        self.data_module = None
        
    def setup_model(self, model_name: str = None, **model_kwargs):
        """设置模型
        
        Args:
            model_name: 模型名称
            **model_kwargs: 模型参数
        """
        if model_name is None:
            model_name = self.config.model_name
        
        # 从装饰器注册器获取模型
        registered_model = decorator_registry.get_model(model_name)
        if not registered_model:
            raise ValueError(f"模型 '{model_name}' 未注册。请使用 @register_model 装饰器注册模型。")
        
        # 使用装饰器注册的模型
        model_class = registered_model['model_class']
        common_step = registered_model['common_step']
        
        # 只传递模型需要的参数
        import inspect
        model_signature = inspect.signature(model_class.__init__)
        model_params = set(model_signature.parameters.keys()) - {'self'}
        
        # 构建模型参数：优先级 model_kwargs > config
        final_kwargs = {}
        for param in model_params:
            if param in model_kwargs:
                final_kwargs[param] = model_kwargs[param]
            elif param in self.config.dict():
                final_kwargs[param] = self.config.dict()[param]
        
        model_instance = model_class(**final_kwargs)
        
        # 使用ModelInterface包装
        self.model = ModelInterface.from_model_and_step(
            model=model_instance,
            common_step=common_step,
            config=self.config.dict()
        )
        
        return self.model
    
    def setup_data(self, dataset_name: str = None, **data_kwargs) -> DataInterface:
        """设置数据模块"""
        if dataset_name is None:
            dataset_name = self.config.dataset_name
        
        # 从装饰器注册器获取数据集
        registered_dataset = decorator_registry.get_dataset(dataset_name)
        if not registered_dataset:
            raise ValueError(f"数据集 '{dataset_name}' 未注册。请使用 @register_dataset 装饰器注册数据集。")
        
        # 使用装饰器注册的数据集
        dataset_class = registered_dataset['dataset_class']
        
        # 只传递DataInterface需要的参数
        import inspect
        data_interface_signature = inspect.signature(DataInterface.__init__)
        data_interface_params = set(data_interface_signature.parameters.keys()) - {'self'}
        
        # 构建DataInterface参数：优先级 data_kwargs > config
        final_kwargs = {}
        for param in data_interface_params:
            if param in data_kwargs:
                final_kwargs[param] = data_kwargs[param]
            elif param in self.config.dict():
                final_kwargs[param] = self.config.dict()[param]
        
        # 添加数据集类
        final_kwargs['dataset_class'] = dataset_class
        final_kwargs['dataset_name'] = dataset_name
        
        # 创建数据模块实例
        self.data_module = DataInterface(**final_kwargs)
        
        return self.data_module
    
    def setup(self, model_name: str = None, dataset_name: str = None, 
              model_kwargs: Dict[str, Any] = None, data_kwargs: Dict[str, Any] = None):
        """设置模型和数据模块
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            model_kwargs: 模型参数（会覆盖配置中的model_kwargs）
            data_kwargs: 数据参数（会覆盖配置中的data_kwargs）
        """
        # 合并参数：方法参数 > 配置参数
        final_model_kwargs = {**self.config.model_kwargs, **(model_kwargs or {})}
        final_data_kwargs = {**self.config.data_kwargs, **(data_kwargs or {})}
        
        self.setup_model(model_name, **final_model_kwargs)
        self.setup_data(dataset_name, **final_data_kwargs)
        
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
        
        return {
            "model_name": self.config.model_name,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
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
