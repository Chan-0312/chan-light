import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


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
    def common_step(module: pl.LightningModule, batch, stage: str = 'train'):
        """训练步骤函数（新的签名：接收 LightningModule 实例）"""
        inputs, labels = batch
        outputs = module.model(inputs)
        
        loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
        accuracy = ((outputs.squeeze() > 0.5).float() == labels.float()).float().mean()
        
        module.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        module.log(f'{stage}_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss


# ========== 可选：定义并绑定生命周期钩子 ==========

def on_fit_start(module: pl.LightningModule):
    print("训练开始！")


def on_train_epoch_start(module: pl.LightningModule):
    print("训练epoch开始！")



def on_train_epoch_end(module: pl.LightningModule):
    print("训练epoch结束！")


# 将钩子绑定到 common_step 上，供 ModelInterface 检测并调用
MyModel.common_step.on_fit_start = on_fit_start
MyModel.common_step.on_train_epoch_start = on_train_epoch_start
MyModel.common_step.on_train_epoch_end = on_train_epoch_end
