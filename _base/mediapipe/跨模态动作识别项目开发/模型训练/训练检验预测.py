# # 在程序开头添加环境检查
# import platform
# print(f"操作系统: {platform.system()} {platform.release()}")
# print(f"Matplotlib后端: {matplotlib.get_backend()}")
# print(f"PySerial版本: {serial.__version__}")
# print(f"PyTorch是否可用GPU: {torch.cuda.is_available()}")

#########################################

print("步骤1：数据转换（直接使用现有CSV）")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')#使用更高效的后端

# 读取已同步数据
data = pd.read_csv('synced_data_6points_n250205.csv')

# 提取肌电窗口（200ms窗口，对应约200行数据）
window_size = 200  # 假设采样率1000Hz
emg_windows = [data['value'].iloc[i:i+window_size].values
              for i in range(0, len(data)-window_size, window_size//2)]  # 50%重叠

# 提取对应时刻的手部关键点坐标（取窗口中间帧）
keypoints = []
for i in range(0, len(data)-window_size, window_size//2):
    mid_idx = i + window_size//2
    kp = data.iloc[mid_idx][[f'x_{n}' for n in range(21)] + [f'y_{n}' for n in range(21)]].values
    keypoints.append(kp.astype(float))

# 保存为PyTorch数据集格式
dataset = {
    'emg': np.array(emg_windows),
    'keypoints': np.array(keypoints)
}
np.savez('handpose_dataset_n250205.npz', **dataset)

##########################################

print("步骤2：简易预测模型搭建")

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
            nn.Linear(256, 21*2)  # 输出21个关键点的x,y坐标
        )

    def forward(self, x):
        return self.fc(x).view(-1, 21, 2)  # 重塑为(21,2)

model = SimpleHandModel()

#############################################

print("步骤3：模型训练（使用现有数据）")

from torch.utils.data import Dataset, DataLoader

class HandDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.emg = torch.FloatTensor(data['emg'])
        self.keypoints = torch.FloatTensor(data['keypoints']).view(-1, 21, 2)

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        return self.emg[idx], self.keypoints[idx]

# 数据加载
dataset = HandDataset('handpose_dataset_n250205.npz')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练配置
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(1500):
    total_loss = 0
    for emg, kp in dataloader:
        optimizer.zero_grad()
        pred = model(emg)
        loss = criterion(pred, kp)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

print("模型训练已完成,正在保存")
torch.save(model.state_dict(), 'hand_model_n250205.pth')

#print("正在进行模型输入输出验证")
# # 在训练后添加测试代码
# test_input = dataset['emg'][0]  # 取第一个样本
# with torch.no_grad():
#     test_output = model(torch.FloatTensor(test_input))
# print("输入信号范围:", test_input.min(), test_input.max())
# print("预测坐标范围 x:", test_output[...,0].min().item(), test_output[...,0].max().item())
# print("预测坐标范围 y:", test_output[...,1].min().item(), test_output[...,1].max().item())

###########################################

print("步骤4.01：配置实时数据采集")

import serial
from collections import deque
import time

# 初始化串口
try:
    ser = serial.Serial(
        port='COM7',        # 修改为实际串口号
        baudrate=115200,    # 需与Arduino程序设置的波特率一致
        timeout=0.1         # 设置读取超时时间
    )
    print("串口已连接:", ser.name)
except Exception as e:  # 使用通用异常捕获
    print("串口连接失败:", e)
    exit()

# 创建环形缓冲区（200个采样点）
buffer = deque(maxlen=200)  # 自动丢弃旧数据

def read_serial():
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:  # 有效数据示例："302.1"
                buffer.append(float(line))
        except UnicodeDecodeError:
            print("解码错误，检查波特率设置")
        except ValueError:
            print("收到非数值数据:", line)

# 测试数据采集
start_time = time.time()
while time.time() - start_time < 5:  # 采集5秒测试数据
    read_serial()
    time.sleep(0.001)  # 防止CPU占用过高

print(f"采集到{len(buffer)}个采样点")
ser.close()  # 测试完成后先关闭

#################################################

print("步骤4.02：整合到可视化程序")

import numpy as np
import matplotlib.pyplot as plt
# 确保正确导入FuncAnimation
from matplotlib.animation import FuncAnimation

def get_real_emg_window():
    """获取最新的200个采样点"""
    while len(buffer) < 200:  # 等待缓冲区填满
        time.sleep(0.01)
    return np.array(list(buffer)[-200:])  # 返回最后200个点

# 初始化画布
fig, ax = plt.subplots()
ax.set_xlim(0, 640)  # 强制设置x轴范围
ax.set_ylim(0, 480)  # 强制设置y轴范围
ax.invert_yaxis()     # 保持坐标系一致

scatter = ax.scatter([], [], s=50, c='red')
lines = [ax.plot([], [], 'b-')[0] for _ in [
    (0,1,2,3,4),          # 拇指
    (0,5,6,7,8),          # 食指
    (0,9,10,11,12),       # 中指
    (0,13,14,15,16),      # 无名指
    (0,17,18,19,20)       # 小指
]]

# 修改update函数
def update(frame):
    read_serial()  # 持续读取新数据

    if len(buffer) >= 200:
        emg_window = get_real_emg_window()

        # 数据预处理（需与训练时一致）
        processed_window = (emg_window - np.mean(emg_window)) / np.std(emg_window)

        # 预测关键点
        with torch.no_grad():
            pred_kp = model(torch.FloatTensor(processed_window).unsqueeze(0))[0].numpy()

        # 坐标转换（假设训练时已归一化）
        pred_kp = pred_kp * np.array([640, 480])  # 还原到原图尺寸

        # 更新可视化元素
        scatter.set_offsets(pred_kp)
        for line, indices in zip(lines, [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),
                                        (0,13,14,15,16),(0,17,18,19,20)]):
            x = [pred_kp[i][0] for i in indices]
            y = [pred_kp[i][1] for i in indices]
            line.set_data(x, y)

        # 添加调试输出
        print("预测坐标示例:", pred_kp[0])  # 打印手腕坐标
        print("x范围:", pred_kp[:, 0].min(), pred_kp[:, 0].max())
        print("y范围:", pred_kp[:, 1].min(), pred_kp[:, 1].max())

    return [scatter] + lines

# 启动动画前打开串口
if not ser.is_open:
    ser.open()
else:
    print("串口已经打开")

# 动态帧数 + 合理缓存
MAX_FRAMES = 1000  # 最大缓存帧数
ani = FuncAnimation(fig, update, frames=None, interval=50, blit=True,
                   cache_frame_data=True, save_count=MAX_FRAMES)

plt.show()
ser.close()  # 窗口关闭后自动断开