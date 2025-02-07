# ---------- 必须在所有matplotlib导入之前设置后端 ----------
import matplotlib

matplotlib.use('TkAgg')  # 使用更高效的后端
# ------------------------------------------------------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import serial
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================== 步骤1：数据转换 ==================
print("步骤1：数据转换")
data = pd.read_csv('backup_database_and_model_repository/synced_data_6points.csv')

# 配置参数
WINDOW_SIZE = 200  # 分析窗口大小（200ms）
SAMPLE_RATE = 1000  # 采样率（Hz）
OVERLAP = WINDOW_SIZE // 2  # 50%重叠

# 提取肌电窗口和关键点
emg_windows = []
keypoints = []
for i in range(0, len(data)-WINDOW_SIZE, OVERLAP):
    mid_idx = i + WINDOW_SIZE//2
    # 正确提取为 (21, 2) 的二维数组
    kp = data.iloc[mid_idx][[f'x_{n}' for n in range(21)] + [f'y_{n}' for n in range(21)]]
    kp = kp.values.reshape(21, 2)  # 关键修改：重塑为二维
    keypoints.append(kp.astype(float))

# 保存数据集
dataset = {
    'emg': np.array(emg_windows),
    'keypoints': np.array(keypoints)  # 现在形状为 (N, 21, 2)
}

np.savez('backup_database_and_model_repository/handpose_dataset.npz', **dataset)

# ================== 步骤2：模型定义 ==================
print("步骤2：模型定义")


class HandPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 21 * 2)  # 输出21个关键点的x,y坐标
        )

    def forward(self, x):
        return self.net(x).view(-1, 21, 2)


model = HandPoseModel()

# ================== 步骤3：模型训练（修正后） ==================
print("步骤3：模型训练")

class HandDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.emg = torch.FloatTensor(data['emg'])
        # 关键点直接加载为 (N, 21, 2)
        self.keypoints = torch.FloatTensor(data['keypoints'])  # 移除了.view()

        # 标准化
        self.emg_mean = self.emg.mean()
        self.emg_std = self.emg.std()
        self.emg = (self.emg - self.emg_mean) / self.emg_std

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        return self.emg[idx], self.keypoints[idx]  # 直接返回三维张量


# 初始化数据集
dataset = HandDataset('backup_database_and_model_repository/handpose_dataset.npz')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 检查第一个样本的形状
# sample_emg, sample_kp = dataset[0]
# print(f"EMG形状: {sample_emg.shape}")  # 应为 (200,)
# print(f"关键点形状: {sample_kp.shape}")  # 应为 (21, 2)

# 训练循环（保持不变）
for epoch in range(100):
    total_loss = 0
    for emg, kp in dataloader:
        torch.optimizer.zero_grad()
        pred = model(emg)

        # 检查形状匹配性
        print(f"Pred shape: {pred.shape}, Target shape: {kp.shape}")  # 调试用

        loss = torch.criterion(pred, kp)
        loss.backward()
        torch.optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(dataloader):.4f}")

# 保存模型
torch.save({
    'model': model.state_dict(),
    'stats': {
        'emg_mean': dataset.emg_mean,
        'emg_std': dataset.emg_std,
        'kp_min': dataset.kp_min,
        'kp_max': dataset.kp_max
    }
}, 'hand_pose_model.pth')

# ================== 步骤4：实时可视化 ==================
print("步骤4：实时可视化")

# 加载模型和统计量
checkpoint = torch.load('hand_pose_model.pth')
model.load_state_dict(checkpoint['model'])
stats = checkpoint['stats']

# 初始化串口
ser = serial.Serial('COM7', 115200, timeout=0.1)
buffer = deque(maxlen=WINDOW_SIZE)

# 初始化画布
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
ax.invert_yaxis()
plt.title("Real-Time Hand Pose Prediction")

# 初始化可视化元素
scatter = ax.scatter([], [], s=40, c='cyan', edgecolors='black')
connections = [
    (0, 1, 2, 3, 4),  # 拇指
    (0, 5, 6, 7, 8),  # 食指
    (0, 9, 10, 11, 12),  # 中指
    (0, 13, 14, 15, 16),  # 无名指
    (0, 17, 18, 19, 20)  # 小指
]
lines = [ax.plot([], [], lw=2, color='orange')[0] for _ in connections]


def get_emg_window():
    """获取预处理后的肌电窗口"""
    while len(buffer) < WINDOW_SIZE:
        time.sleep(0.001)
    window = np.array(list(buffer)[-WINDOW_SIZE:])
    return (window - stats['emg_mean'].item()) / stats['emg_std'].item()


def update(frame):
    # 读取串口数据
    while ser.in_waiting > 0:
        try:
            value = float(ser.readline().decode().strip())
            buffer.append(value)
        except:
            pass

    if len(buffer) >= WINDOW_SIZE:
        # 数据预处理
        processed_window = get_emg_window()

        # 预测关键点
        with torch.no_grad():
            pred_kp = model(torch.FloatTensor(processed_window).unsqueeze(0))[0].numpy()

        # 反归一化
        pred_kp = pred_kp * (stats['kp_max'] - stats['kp_min']).numpy() + stats['kp_min'].numpy()

        # 更新可视化
        scatter.set_offsets(pred_kp)
        for line, indices in zip(lines, connections):
            x = [pred_kp[i, 0] for i in indices]
            y = [pred_kp[i, 1] for i in indices]
            line.set_data(x, y)

        # 动态调整视图
        ax.set_xlim(pred_kp[:, 0].min() - 50, pred_kp[:, 0].max() + 50)
        ax.set_ylim(pred_kp[:, 1].min() - 50, pred_kp[:, 1].max() + 50)

    return [scatter] + lines


# 启动动画
ani = FuncAnimation(
    fig, update,
    interval=50,  # 每50ms更新一次 (约20fps)
    blit=True,
    cache_frame_data=False
)

try:
    plt.show()
finally:
    ser.close()
    print("串口已安全关闭")