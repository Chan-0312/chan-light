"""
模型包装器 - 将用户模型包装为PyTorch Lightning模块
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Callable, Optional, Any, Dict


class ModelInterface(pl.LightningModule):
    """模型接口，将用户的模型包装为PyTorch Lightning模块"""
    
    def __init__(self, 
                 model_class: type,
                 common_step: Callable,
                 model_kwargs: Dict[str, Any] = None,
                 optimizer: str = 'adam',
                 lr: float = 0.001,
                 weight_decay: float = 1e-5,
                 lr_scheduler: str = 'cosine',
                 lr_decay_steps: int = 20,
                 lr_decay_rate: float = 0.5,
                 lr_decay_min_lr: float = 1e-5,
                 epochs: int = 50,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class', 'common_step'])
        
        # 在内部创建模型实例
        self.model = self.instancialize(model_class, model_kwargs or {})
        self.common_step = common_step
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        return self.common_step(self.model, batch, self.log, self.hparams, stage='train')
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        return self.common_step(self.model, batch, self.log, self.hparams, stage='val')
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.common_step(self.model, batch, self.log, self.hparams, stage='test')
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 根据选择的优化器类型初始化优化器
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f'不支持的优化器类型: {self.hparams.optimizer}')

        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == 'None':
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate
                )
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.epochs,
                    eta_min=self.hparams.lr_decay_min_lr
                )
            else:
                raise ValueError(f'不支持的学习率调度器类型: {self.hparams.lr_scheduler}')
            return [optimizer], [scheduler]
    
    @staticmethod
    def instancialize(model_class: type, model_kwargs: Dict[str, Any]) -> nn.Module:
        """创建模型实例
        
        Args:
            model_class: 模型类
            model_kwargs: 模型参数字典
            
        Returns:
            nn.Module: 模型实例
        """
        import inspect
        
        # 获取模型类的构造函数参数（排除 self）
        class_args = list(inspect.signature(model_class.__init__).parameters.keys())[1:]
        
        # 构建参数字典：只传递模型类需要的参数
        args_dict = {}
        for arg in class_args:
            if arg in model_kwargs:
                args_dict[arg] = model_kwargs[arg]
        
        return model_class(**args_dict)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model.__class__.__name__ if self.model else "Unknown",
            "total_params": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0,
            "model_kwargs": self.hparams.model_kwargs
        }