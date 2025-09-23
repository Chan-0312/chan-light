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


class TrainingManager:
    """训练管理器，提供高级训练接口"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.trainer = PLTrainer(config)
        self.model = None
        self.data_module = None
        
    def setup_model(self, model_or_path=None, model_kwargs: Dict[str, Any] = None, 
                   common_step=None):
        """设置模型
        
        Args:
            model_or_path: 模型类（直接传入）或模型文件路径（字符串）
            model_kwargs: 模型参数
            common_step: 训练步骤函数（可选，当直接传入模型类时需要）
        """
        if model_or_path is None:
            model_or_path = self.config.model_or_path
        
        # 判断 model_or_path 是模型类还是文件路径字符串
        if isinstance(model_or_path, type) or callable(model_or_path):
            # 方式1：直接传入模型类
            model_class = model_or_path
            if common_step is None:
                # 尝试从模型类获取 common_step
                common_step = getattr(model_class, 'common_step', None)
                if not common_step or not callable(common_step):
                    raise ValueError(f"模型类 {model_class.__name__} 必须定义 common_step 方法，或者传入 common_step 参数")
        else:
            # 方式2：字符串路径，从文件导入模型
            try:
                model_class, common_step = self._import_model_from_file(model_or_path)
            except Exception as e:
                raise ValueError(f"无法从文件 '{model_or_path}' 导入模型: {str(e)}\n"
                               f"请确保文件路径正确，且包含模型类和 common_step 函数")
        
        # 创建模型接口（内部会自动创建模型实例）
        config_dict = self.config.dict()
        config_dict['model_kwargs'] = model_kwargs  # 确保使用正确的 model_kwargs
        self.model = ModelInterface(
            model_class=model_class,
            common_step=common_step,
            **config_dict
        )
        
        return self.model
    
    def _import_model_from_file(self, file_path):
        """从文件路径导入模型
        
        基于文件名推导类名：文件名 snake_case.py -> 类名 CamelCase
        
        Args:
            file_path: 模型文件路径（支持带或不带 .py 扩展名）
            
        Returns:
            tuple: (model_class, common_step)
        """
        import importlib.util
        import sys
        from pathlib import Path
        
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
        
        # 查找 common_step 函数
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
            raise ValueError(f"在文件 {file_path} 中未找到 common_step 函数")
        
        return model_class, common_step
    
    def setup_data(self, dataset_or_path=None, data_kwargs: Dict[str, Any] = None) -> DataInterface:
        """设置数据模块
        
        Args:
            dataset_or_path: 数据集类（直接传入）或数据集文件路径（字符串）
            data_kwargs: 数据参数
        """
        if dataset_or_path is None:
            dataset_or_path = self.config.dataset_or_path
        
        # 判断 dataset_or_path 是数据集类还是文件路径字符串
        if isinstance(dataset_or_path, type) or callable(dataset_or_path):
            # 方式1：直接传入数据集类
            dataset_class = dataset_or_path
        else:
            # 方式2：字符串路径，从文件导入数据集
            try:
                dataset_class = self._import_dataset_from_file(dataset_or_path)
            except Exception as e:
                raise ValueError(f"无法从文件 '{dataset_or_path}' 导入数据集: {str(e)}\n"
                               f"请确保文件路径正确，且包含数据集类")
        
        # 创建数据模块实例
        config_dict = self.config.dict()
        config_dict['data_kwargs'] = data_kwargs  # 确保使用正确的 data_kwargs
        self.data_module = DataInterface(
            dataset_class=dataset_class,
            **config_dict
        )
        
        return self.data_module
    
    def _import_dataset_from_file(self, file_path):
        """从文件路径导入数据集
        
        基于文件名推导类名：文件名 snake_case.py -> 类名 CamelCase
        
        Args:
            file_path: 数据集文件路径（支持带或不带 .py 扩展名）
            
        Returns:
            class: dataset_class
        """
        import importlib.util
        import sys
        from pathlib import Path
        
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
        dataset_class = None
        
        # 从文件名推导类名：snake_case.py -> CamelCase
        file_name = file_path.stem  # 去掉 .py 扩展名
        camel_name = ''.join([i.capitalize() for i in file_name.split('_')])
        
        try:
            # 尝试获取推导出的类名
            dataset_class = getattr(module, camel_name)
            if not (isinstance(dataset_class, type) and 
                    hasattr(dataset_class, '__init__') and 
                    hasattr(dataset_class, '__len__') and 
                    hasattr(dataset_class, '__getitem__')):
                dataset_class = None
                raise AttributeError(f"类 {camel_name} 不是有效的数据集类")
            
            return dataset_class
            
        except (AttributeError, ValueError):
            # 如果基于文件名推导失败，直接抛出错误
            raise ValueError(f"在文件 {file_path} 中未找到数据集类。\n"
                           f"请确保文件包含数据集类，并按照约定命名：\n"
                           f"- 文件名：snake_case.py\n"
                           f"- 类名：{camel_name}")
        
        return dataset_class
    
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
        
        self.setup_model(model_or_path, final_model_kwargs, common_step)
        self.setup_data(dataset_or_path, final_data_kwargs)
        
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
