####################################（1）定义 PyTorch 模型
import torch
import torch.nn as nn
import paddle
import numpy as np

import os
os.environ["OMP_NUM_THREADS"] = "1"

class HandKeypointsModelTorch(nn.Module):
    def __init__(self, max_lines=6, max_points=5):
        super(HandKeypointsModelTorch, self).__init__()
        self.max_lines = max_lines
        self.max_points = max_points
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),  # 根据输入图片大小调整
            nn.ReLU(),
            nn.Linear(512, max_lines * max_points * 2),  # 动态输出维度
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


################################################################（2）加载飞桨模型权重并转换为 PyTorch 模型


# 加载飞桨模型权重
paddle_state_dict = paddle.load(r"YOLOv11_HandPose/Format_Conversion/onnx2pytorch/hand_keypoints_model.pdparams")

# 将飞桨模型权重转换为 PyTorch 格式
torch_state_dict = {}
for key, value in paddle_state_dict.items():
    # 将飞桨的 Tensor 转换为 NumPy 数组，再转换为 PyTorch 的 Tensor
    torch_state_dict[key] = torch.from_numpy(np.array(value))

# 创建 PyTorch 模型并加载权重
model_torch = HandKeypointsModelTorch(max_lines=6, max_points=5)
model_torch.load_state_dict(torch_state_dict)

# 保存为 PyTorch 的 .pt 文件
torch.save(model_torch.state_dict(), r"YOLOv11_HandPose/Format_Conversion/onnx2pytorch/hand_keypoints_model.pt")