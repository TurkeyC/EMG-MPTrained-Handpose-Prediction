import pandas as pd
import numpy as np

# 读取已同步数据
data = pd.read_csv('synced_data_6points.csv')

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
np.savez('handpose_dataset.npz', **dataset)