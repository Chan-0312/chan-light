import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """数据集"""
    
    def __init__(self, data_size=1000, input_dim=1024, stage='train'):
        self.data_size = data_size
        self.input_dim = input_dim
        self.stage = stage
        
        # 生成模拟数据
        self.data = torch.randn(data_size, input_dim)
        self.labels = torch.randint(0, 2, (data_size,))
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
