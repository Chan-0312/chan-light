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
                 model: nn.Module,
                 common_step: Callable,
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
        self.save_hyperparameters(ignore=['model', 'common_step'])
        
        self.model = model
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
    
    @classmethod
    def from_model_and_step(cls, 
                           model: nn.Module, 
                           common_step: Callable,
                           config: Optional[Dict[str, Any]] = None) -> 'ModelInterface':
        """从模型和步骤函数创建接口"""
        config = config or {}
        return cls(model=model, common_step=common_step, **config)
