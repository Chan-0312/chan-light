"""
数据接口 - 基于原始data_interface.py的SDK版本
"""

import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional


class DataInterface(pl.LightningDataModule):
    """数据接口类，支持动态加载数据集"""

    def __init__(self, 
                 dataset_class=None,
                 num_workers: int = 8,
                 dataset_name: str = '',
                 batch_size: int = 32,
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset_name
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.dataset_class = dataset_class
        self.load_data_module()

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 为训练和验证分配数据集
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=True)
            self.valset = self.instancialize(train=False)

        # 为测试分配数据集
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(train=False)
    
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

    def load_data_module(self):
        """动态加载数据模块"""
        if self.dataset_class is not None:
            # 使用装饰器注册的数据集类
            self.data_module = self.dataset_class
        else:
            # 传统方式：从模块加载
            name = self.dataset
            # 将 snake_case.py 文件名转换为 CamelCase 类名
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
            try:
                self.data_module = getattr(importlib.import_module(
                    '.'+name, package=__package__), camel_name)
            except Exception as e:
                raise ValueError(
                    f'无法加载数据集模块 data.{name}.{camel_name}: {str(e)}')

    def instancialize(self, **other_args):
        """使用配置参数实例化数据集"""
        class_args = list(inspect.signature(self.data_module).parameters.keys())
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            "dataset_name": self.dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "train_size": len(self.trainset) if hasattr(self, 'trainset') else 0,
            "val_size": len(self.valset) if hasattr(self, 'valset') else 0,
            "test_size": len(self.testset) if hasattr(self, 'testset') else 0
        }
