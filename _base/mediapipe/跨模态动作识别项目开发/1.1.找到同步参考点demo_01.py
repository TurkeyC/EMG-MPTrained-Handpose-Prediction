# 可视化肌电信号（运行后截图确认同步点）
import pandas as pd
import matplotlib.pyplot as plt

emg = pd.read_csv('E:\\User_Stuff\\Documents\\Code_Typing\\Python_3.12_Code_Workspace\\SmallProject\\YOLO_handpose\\mediapipe\\跨模态动作识别项目开发\\emg_data.csv')
plt.figure(figsize=(12,4))
plt.plot(emg['Column2'])
plt.title('EMG信号全览（寻找拍手尖峰）')
plt.show()