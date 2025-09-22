"""
日志工具
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "chanlight", level: int = logging.INFO) -> logging.Logger:
    """获取日志记录器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(console_handler)
    
    return logger


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """设置日志配置"""
    logger = get_logger(level=level)
    
    if log_file:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加文件处理器
        logger.addHandler(file_handler)
    
    return logger
