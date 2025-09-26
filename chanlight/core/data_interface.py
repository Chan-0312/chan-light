"""
数据接口 - 基于原始data_interface.py的SDK版本
"""

import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Type
from pathlib import Path
import importlib.util
import sys
from chanlight.core.config import TrainerConfig


class DataInterface(pl.LightningDataModule):
    """数据接口类，支持动态加载数据集"""

    def __init__(self,
                 dataset_class,
                 data_kwargs: Dict[str, Any] = {},
                 num_workers: int = 8,
                 batch_size: int = 32,
                 **kwargs):
        """初始化数据接口
        
        Args:
            dataset_class: 数据集类
            num_workers: 数据加载器的工作进程数
            batch_size: 批次大小
            **kwargs: 其他参数，会传递给数据集类的构造函数
        """
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_kwargs = data_kwargs
        self.dataset_class = dataset_class
        self.data_module = dataset_class

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 为训练和验证分配数据集
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(stage='train')
            self.valset = self.instancialize(stage='val')

        # 为测试分配数据集
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(stage='test')
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.valset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        return DataLoader(
            self.testset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )

    def instancialize(self, **other_args):
        """使用配置参数实例化数据集"""
        class_args = list(inspect.signature(self.data_module).parameters.keys())
        inkeys = self.data_kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.data_kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            "dataset_name": self.dataset_class.__name__ if self.dataset_class else "Unknown",
            "data_kwargs": self.data_kwargs,
        }



def setup_data_interface(dataset_or_path, data_kwargs: Dict[str, Any] = None, 
              config: TrainerConfig = None) -> DataInterface:
    """设置数据模块
    
    Args:
        dataset_or_path: 数据集类（直接传入）或数据集文件路径（字符串）
        data_kwargs: 数据参数
        config: 训练配置
        
    Returns:
        DataInterface: 数据接口实例
    """
    if dataset_or_path is None:
        dataset_or_path = config.dataset_or_path
    
    # 判断 dataset_or_path 是数据集类还是文件路径字符串
    if isinstance(dataset_or_path, type) or callable(dataset_or_path):
        # 方式1：直接传入数据集类
        dataset_class = dataset_or_path
    else:
        # 方式2：字符串路径，从文件导入数据集
        dataset_class = _import_dataset_from_file(dataset_or_path)
    
    # 创建数据模块实例
    config_dict = config.dict() if config else {}
    config_dict['data_kwargs'] = data_kwargs  # 确保使用正确的 data_kwargs
    return DataInterface(
        dataset_class=dataset_class,
        **config_dict
    )





def _import_dataset_from_file(file_path) -> Type:
    """从文件路径导入数据集
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        Type: 数据集类
    """
    # 处理文件路径
    file_path = Path(file_path)
    if not file_path.exists():
        # 尝试添加 .py 扩展名
        py_path = file_path.with_suffix('.py')
        if py_path.exists():
            file_path = py_path
        else:
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("dataset_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dataset_module"] = module
    spec.loader.exec_module(module)
    
    # 基于文件名推导类名：snake_case.py -> CamelCase
    file_name = file_path.stem
    camel_name = ''.join([i.capitalize() for i in file_name.split('_')])
    
    try:
        dataset_class = getattr(module, camel_name)
        if not (isinstance(dataset_class, type) and 
                hasattr(dataset_class, '__init__') and 
                hasattr(dataset_class, '__len__') and 
                hasattr(dataset_class, '__getitem__')):
            dataset_class = None
            raise AttributeError(f"类 {camel_name} 不是有效的数据集类")
    except (AttributeError, ValueError):
        # 如果基于文件名推导失败，直接抛出错误
        raise ValueError(f"在文件 {file_path} 中未找到数据集类。\n"
                         f"请确保文件包含数据集类，并按照约定命名：\n"
                         f"- 文件名：snake_case.py\n"
                         f"- 类名：{camel_name}")
    
    return dataset_class