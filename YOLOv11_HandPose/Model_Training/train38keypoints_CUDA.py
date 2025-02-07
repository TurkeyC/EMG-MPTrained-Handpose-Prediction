import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 定义数据集类
class HandKeypointsDataset(Dataset):
    def __init__(self, image_dir, label_dir, max_lines=6, max_points=5):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.max_lines = max_lines
        self.max_points = max_points
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图片
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # 调整图片大小为 224x224
        image = image / 255.0  # 归一化

        # 读取标签
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 解析标签
        keypoints = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            keypoints.extend(parts[1:])  # 跳过类别ID

        # 填充关键点
        keypoints = self.pad_keypoints(keypoints, self.max_lines, self.max_points)

        # 转换为 Tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return image, keypoints

    def pad_keypoints(self, keypoints, max_lines, max_points):
        # 将关键点填充为固定格式
        padded_keypoints = []
        for i in range(max_lines):
            start = i * max_points * 2
            end = start + max_points * 2
            line_keypoints = keypoints[start:end] if start < len(keypoints) else []
            if len(line_keypoints) < max_points * 2:
                line_keypoints += [0.0, 0.0] * (max_points - len(line_keypoints) // 2)
            padded_keypoints.extend(line_keypoints)
        return padded_keypoints

# 定义模型
class HandKeypointsModel(nn.Module):
    def __init__(self, max_lines=6, max_points=5):
        super(HandKeypointsModel, self).__init__()
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

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建数据集
image_dir = r"YOLOv11_HandPose/Model_Training/Hand_Keypoints_Dataset/Images_Data/train"
label_dir = r"YOLOv11_HandPose/Model_Training/Hand_Keypoints_Dataset/Labels_Data/train"
dataset = HandKeypointsDataset(image_dir, label_dir, max_lines=6, max_points=5)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 创建模型并移动到 GPU
model = HandKeypointsModel(max_lines=6, max_points=5).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for images, keypoints in dataloader:
        # 将数据移动到 GPU
        images = images.to(device)
        keypoints = keypoints.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, keypoints)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# 保存模型
torch.save(model.state_dict(), r"YOLOv11_HandPose/Model_Training/Model_Output/hand_keypoints_model_bycuda.pt")