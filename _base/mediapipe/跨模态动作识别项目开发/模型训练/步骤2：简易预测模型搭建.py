import torch
import torch.nn as nn


class SimpleHandModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(200, 128),  # 输入200个肌电采样点
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 21 * 2)  # 输出21个关键点的x,y坐标
        )

    def forward(self, x):
        return self.fc(x).view(-1, 21, 2)  # 重塑为(21,2)


model = SimpleHandModel()