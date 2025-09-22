"""
装饰器系统 - 用于注册模型和数据集
"""

from typing import Dict, Type, Callable, Any, Optional
import functools


class Registry:
    """全局注册器"""
    
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._datasets: Dict[str, Dict[str, Any]] = {}
    
    def register_model(self, name: str, model_class: Type, common_step: Callable, **kwargs):
        """注册模型"""
        self._models[name] = {
            'model_class': model_class,
            'common_step': common_step,
            'kwargs': kwargs
        }
    
    def register_dataset(self, name: str, dataset_class: Type, **kwargs):
        """注册数据集"""
        self._datasets[name] = {
            'dataset_class': dataset_class,
            'kwargs': kwargs
        }
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模型"""
        return self._models.get(name)
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """获取数据集"""
        return self._datasets.get(name)
    
    def list_models(self) -> list:
        """列出所有模型"""
        return list(self._models.keys())
    
    def list_datasets(self) -> list:
        """列出所有数据集"""
        return list(self._datasets.keys())


# 全局注册器实例
registry = Registry()


def register_model(name: str, **model_kwargs):
    """
    模型注册装饰器
    
    Args:
        name: 模型名称
        **model_kwargs: 模型默认参数
    """
    def decorator(model_class):
        # 获取common_step函数
        common_step = getattr(model_class, 'common_step', None)
        if common_step is None:
            raise ValueError(f"模型 {name} 必须定义 common_step 方法")
        
        # 注册模型
        registry.register_model(name, model_class, common_step, **model_kwargs)
        
        # 添加注册信息到模型类
        model_class._chanlight_name = name
        model_class._chanlight_registered = True
        
        return model_class
    
    return decorator


def register_dataset(name: str, **dataset_kwargs):
    """
    数据集注册装饰器
    
    Args:
        name: 数据集名称
        **dataset_kwargs: 数据集默认参数
    """
    def decorator(dataset_class):
        # 注册数据集
        registry.register_dataset(name, dataset_class, **dataset_kwargs)
        
        # 添加注册信息到数据集类
        dataset_class._chanlight_name = name
        dataset_class._chanlight_registered = True
        
        return dataset_class
    
    return decorator




def get_registry() -> Registry:
    """获取全局注册器"""
    return registry
