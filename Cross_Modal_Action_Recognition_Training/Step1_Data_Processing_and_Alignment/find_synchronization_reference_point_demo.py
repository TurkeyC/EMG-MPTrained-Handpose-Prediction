# 可视化肌电信号（运行后截图确认同步点）
import pandas as pd
import matplotlib.pyplot as plt

emg = pd.read_csv(r"Cross_Modal_Action_Recognition_Training/Backup_Database_and_Model_Repository_TOP/Emg_Data/emg_data.csv")
plt.figure(figsize=(12,4))
plt.plot(emg['Column2'])
plt.title('EMG信号全览（寻找拍手尖峰）')
plt.show()