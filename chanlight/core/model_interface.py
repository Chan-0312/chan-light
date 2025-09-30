"""
模型包装器 - 将用户模型包装为PyTorch Lightning模块
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Callable, Optional, Any, Dict, Tuple, Type
from pathlib import Path
import importlib.util
import sys
from chanlight.core.config import TrainerConfig


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
        return self.common_step(self, batch, stage='train')
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        return self.common_step(self, batch, stage='val')
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.common_step(self, batch, stage='test')
    
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
            "model_kwargs": self.hparams.model_kwargs
        }
    
    # ========== PyTorch Lightning 生命周期钩子 ==========
    
    def on_train_epoch_start(self):
        """训练epoch开始时调用"""
        if hasattr(self.common_step, 'on_train_epoch_start'):
            self.common_step.on_train_epoch_start(self)
    
    def on_train_epoch_end(self):
        """训练epoch结束时调用"""
        if hasattr(self.common_step, 'on_train_epoch_end'):
            self.common_step.on_train_epoch_end(self)
    
    def on_validation_epoch_start(self):
        """验证epoch开始时调用"""
        if hasattr(self.common_step, 'on_validation_epoch_start'):
            self.common_step.on_validation_epoch_start(self)
    
    def on_validation_epoch_end(self):
        """验证epoch结束时调用"""
        if hasattr(self.common_step, 'on_validation_epoch_end'):
            self.common_step.on_validation_epoch_end(self)
    
    def on_test_epoch_start(self):
        """测试epoch开始时调用"""
        if hasattr(self.common_step, 'on_test_epoch_start'):
            self.common_step.on_test_epoch_start(self)
    
    def on_test_epoch_end(self):
        """测试epoch结束时调用"""
        if hasattr(self.common_step, 'on_test_epoch_end'):
            self.common_step.on_test_epoch_end(self)
    
    def on_train_batch_start(self, batch, batch_idx):
        """训练batch开始时调用"""
        if hasattr(self.common_step, 'on_train_batch_start'):
            return self.common_step.on_train_batch_start(self, batch, batch_idx)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """训练batch结束时调用"""
        if hasattr(self.common_step, 'on_train_batch_end'):
            return self.common_step.on_train_batch_end(self, outputs, batch, batch_idx)
    
    def on_validation_batch_start(self, batch, batch_idx):
        """验证batch开始时调用"""
        if hasattr(self.common_step, 'on_validation_batch_start'):
            return self.common_step.on_validation_batch_start(self, batch, batch_idx)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """验证batch结束时调用"""
        if hasattr(self.common_step, 'on_validation_batch_end'):
            return self.common_step.on_validation_batch_end(self, outputs, batch, batch_idx)
    
    def on_test_batch_start(self, batch, batch_idx):
        """测试batch开始时调用"""
        if hasattr(self.common_step, 'on_test_batch_start'):
            return self.common_step.on_test_batch_start(self, batch, batch_idx)
    
    def on_test_batch_end(self, outputs, batch, batch_idx):
        """测试batch结束时调用"""
        if hasattr(self.common_step, 'on_test_batch_end'):
            return self.common_step.on_test_batch_end(self, outputs, batch, batch_idx)
    
    def on_before_zero_grad(self, optimizer):
        """梯度清零前调用"""
        if hasattr(self.common_step, 'on_before_zero_grad'):
            return self.common_step.on_before_zero_grad(self, optimizer)
    
    def on_after_backward(self):
        """反向传播后调用"""
        if hasattr(self.common_step, 'on_after_backward'):
            return self.common_step.on_after_backward(self)
    
    def on_before_optimizer_step(self, optimizer):
        """优化器步骤前调用"""
        if hasattr(self.common_step, 'on_before_optimizer_step'):
            return self.common_step.on_before_optimizer_step(self, optimizer)
    
    def on_after_optimizer_step(self, optimizer):
        """优化器步骤后调用"""
        if hasattr(self.common_step, 'on_after_optimizer_step'):
            return self.common_step.on_after_optimizer_step(self, optimizer)
    
    def on_fit_start(self):
        """训练开始时调用"""
        if hasattr(self.common_step, 'on_fit_start'):
            return self.common_step.on_fit_start(self)
    
    def on_fit_end(self):
        """训练结束时调用"""
        if hasattr(self.common_step, 'on_fit_end'):
            return self.common_step.on_fit_end(self)
    
    def on_save_checkpoint(self, checkpoint):
        """保存检查点时调用"""
        if hasattr(self.common_step, 'on_save_checkpoint'):
            return self.common_step.on_save_checkpoint(self, checkpoint)
    
    def on_load_checkpoint(self, checkpoint):
        """加载检查点时调用"""
        if hasattr(self.common_step, 'on_load_checkpoint'):
            return self.common_step.on_load_checkpoint(self, checkpoint)



def setup_model_interface(model_or_path, model_kwargs: Dict[str, Any] = None, 
               common_step:Callable=None, config: TrainerConfig = None, 
               ckpt_path: str = None) -> ModelInterface:
    """设置模型
    
    Args:
        model_or_path: 模型类（直接传入）或模型文件路径（字符串）
        model_kwargs: 模型参数
        common_step: 通用步骤函数
        config: 训练配置
        ckpt_path: 检查点文件路径（可选）
        
    Returns:
        ModelInterface: 模型接口实例
    """
    if model_or_path is None:
        model_or_path = config.model_or_path
    
    # 判断 model_or_path 是模型类还是文件路径字符串
    if isinstance(model_or_path, type) or callable(model_or_path):
        # 方式1：直接传入模型类
        model_class = model_or_path
        # 允许省略 common_step：若类上定义了 common_step 则自动获取
        if common_step is None and hasattr(model_class, 'common_step') and callable(getattr(model_class, 'common_step')):
            common_step = getattr(model_class, 'common_step')
        if common_step is None:
            raise ValueError("直接传入模型类时，必须提供 common_step，或在模型类上定义静态方法 common_step(module, batch, stage='train')")
    else:
        # 方式2：字符串路径，从文件导入模型
        model_class, common_step = _import_model_from_file(model_or_path)
    
    # 创建模型接口实例
    config_dict = config.dict() if config else {}
    config_dict['model_kwargs'] = model_kwargs  # 确保使用正确的 model_kwargs
    
    model_interface = ModelInterface(
        model_class=model_class,
        common_step=common_step,
        **config_dict
    )
    
    # 如果提供了检查点路径，加载检查点
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model_interface.load_state_dict(checkpoint['state_dict'])
        print(f"已从检查点加载模型: {ckpt_path}")
    
    return model_interface


def _import_model_from_file(file_path) -> Tuple[Type, Callable]:
    """从文件路径导入模型
    
    Args:
        file_path: 模型文件路径
        
    Returns:
        tuple: (model_class, common_step)
    """
    # 如果是相对路径，转换为绝对路径
    file_path = Path(file_path).resolve()
    
    # 如果文件不存在且没有 .py 扩展名，尝试添加 .py 扩展名
    if not file_path.exists() and file_path.suffix != '.py':
        py_file_path = file_path.with_suffix('.py')
        if py_file_path.exists():
            file_path = py_file_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if file_path.suffix != '.py':
        raise ValueError(f"文件必须是 .py 文件: {file_path}")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)
    
    # 方式1：基于文件名推导类名（推荐方式）
    model_class = None
    common_step = None
    
    # 从文件名推导类名：snake_case.py -> CamelCase
    file_name = file_path.stem  # 去掉 .py 扩展名
    camel_name = ''.join([i.capitalize() for i in file_name.split('_')])

    try:
        # 尝试获取推导出的类名
        model_class = getattr(module, camel_name)
        if not (isinstance(model_class, type) and 
                hasattr(model_class, '__init__') and 
                hasattr(model_class, 'forward')):
            model_class = None
            raise AttributeError(f"类 {camel_name} 不是有效的模型类")
        
    except (AttributeError, ValueError):
        # 如果基于文件名推导失败，直接抛出错误
        raise ValueError(f"在文件 {file_path} 中未找到模型类。\n"
                        f"请确保文件包含模型类，并按照约定命名：\n"
                        f"- 文件名：snake_case.py\n"
                        f"- 类名：{camel_name}")
    
    # 新的 common_step 签名: @staticmethod def common_step(module: pl.LightningModule, batch, stage: str = 'train')
    if hasattr(model_class, 'common_step') and callable(model_class.common_step):
        common_step = model_class.common_step
    else:
        # 尝试获取独立的 common_step 函数
        try:
            common_step = getattr(module, 'common_step')
        except AttributeError:
            # 最后尝试模糊匹配包含 common_step 的函数
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (callable(attr) and 
                    attr_name.lower().find('common_step') != -1):
                    common_step = attr
                    break
    
    if not common_step:
        raise ValueError(f"在文件 {file_path} 中未找到 common_step 函数。\n"
                        f"请确保定义了 common_step 函数，签名如下：\n"
                        f"@staticmethod\n"
                        f"def common_step(module: pl.LightningModule, batch, stage: str = 'train'):\n"
                        f"    # 提取数据\n"
                        f"    inputs, labels = batch\n"
                        f"    # 你的训练逻辑...")
    
    return model_class, common_step