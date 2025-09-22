"""
PyTorch Lightning 训练器封装
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, List, Dict, Any
from pathlib import Path

from chanlight.core.config import TrainerConfig


class PLTrainer:
    """PyTorch Lightning 训练器封装类"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.trainer = None
        self.model = None
        self.data_module = None
        
    def setup(self, model, data_module):
        """设置模型和数据模块"""
        self.model = model
        self.data_module = data_module
        
        # 设置随机种子
        pl.seed_everything(self.config.seed, verbose=False)
        
        # 设置设备精度
        torch.set_float32_matmul_precision('high')
        
    def _create_callbacks(self) -> List[plc.Callback]:
        """创建训练回调"""
        callbacks = []
        
        # 模型检查点
        filename = f'checkpoint-{{epoch:02d}}-{{{self.config.monitor_metric}:.3f}}'
        callbacks.append(plc.ModelCheckpoint(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            filename=filename,
            save_top_k=3,
            save_last=True
        ))
        
        # 学习率监控器
        callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))
        
        # 早停
        if self.config.use_early_stopping:
            callbacks.append(plc.EarlyStopping(
                monitor=self.config.monitor_metric,
                mode=self.config.monitor_mode,
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta
            ))
        
        # SWA
        if self.config.use_swa:
            callbacks.append(plc.StochasticWeightAveraging(
                swa_lrs=self.config.swa_lrs,
                swa_epoch_start=self.config.swa_epoch_start
            ))
        
        return callbacks
    
    def _create_logger(self) -> TensorBoardLogger:
        """创建日志记录器"""
        return TensorBoardLogger(
            save_dir=self.config.log_dir,
            name=self.config.model_name
        )
    
    def _create_trainer(self) -> Trainer:
        """创建PyTorch Lightning训练器"""
        callbacks = self._create_callbacks()
        logger = self._create_logger()
        
        # 设备配置
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                devices = 'auto'
                accelerator = 'gpu'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices = 'auto'
                accelerator = 'mps'
            else:
                devices = 1
                accelerator = 'cpu'
        else:
            devices = 1
            accelerator = self.config.device
        
        return Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=self.config.epochs,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            devices=devices,
            accelerator=accelerator,
            precision=self.config.precision
        )
    
    def train(self, ckpt_path: Optional[str] = None) -> Dict[str, Any]:
        """开始训练"""
        if self.model is None or self.data_module is None:
            raise ValueError("请先调用 setup() 方法设置模型和数据模块")
        
        self.trainer = self._create_trainer()
        
        # 训练模型
        self.trainer.fit(
            self.model, 
            datamodule=self.data_module,
            ckpt_path=ckpt_path
        )
        
        return {
            "best_model_path": self.trainer.checkpoint_callback.best_model_path,
            "last_model_path": self.trainer.checkpoint_callback.last_model_path,
            "best_model_score": self.trainer.checkpoint_callback.best_model_score
        }
    
    def test(self, ckpt_path: Optional[str] = None) -> Dict[str, Any]:
        """测试模型"""
        if self.trainer is None:
            self.trainer = self._create_trainer()
        
        if ckpt_path is None:
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
        
        results = self.trainer.test(
            self.model,
            datamodule=self.data_module,
            ckpt_path=ckpt_path
        )
        
        return results[0] if results else {}
    
    def predict(self, dataloader, ckpt_path: Optional[str] = None) -> List[Any]:
        """预测"""
        if self.trainer is None:
            self.trainer = self._create_trainer()
        
        if ckpt_path is None:
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
        
        predictions = self.trainer.predict(
            self.model,
            dataloaders=dataloader,
            ckpt_path=ckpt_path
        )
        
        return predictions
    
    def get_model(self) -> pl.LightningModule:
        """获取模型"""
        return self.model
    
    def get_data_module(self) -> pl.LightningDataModule:
        """获取数据模块"""
        return self.data_module
