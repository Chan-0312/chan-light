import torch
import torch.nn as nn
from torch.nn import functional as F


class MyModel(nn.Module):
    """CNN模型"""
    
    def __init__(self, input_size=1024, hidden_size=256, num_classes=1, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

    @staticmethod
    def common_step(model, batch, log, hparams, stage='train'):
        """训练步骤函数"""
        inputs, labels = batch
        outputs = model(inputs)
        
        loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
        accuracy = ((outputs.squeeze() > 0.5).float() == labels.float()).float().mean()
        
        log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        log(f'{stage}_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
