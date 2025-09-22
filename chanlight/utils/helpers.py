"""
辅助工具函数
"""

import os
from pathlib import Path
from typing import List, Union


def create_directories(directories: Union[str, List[str]]) -> None:
    """创建目录"""
    if isinstance(directories, str):
        directories = [directories]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def ensure_directory_exists(directory: str) -> None:
    """确保目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f}{size_names[i]}"


def count_parameters(model) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """计算可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model) -> dict:
    """获取模型摘要"""
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }
