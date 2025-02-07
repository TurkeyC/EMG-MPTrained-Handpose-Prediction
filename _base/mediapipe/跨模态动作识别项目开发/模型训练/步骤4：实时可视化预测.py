import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化画布
fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=50, c='red')
lines = [ax.plot([], [], 'b-')[0] for _ in [
    (0, 1, 2, 3, 4),  # 拇指
    (0, 5, 6, 7, 8),  # 食指
    (0, 9, 10, 11, 12),  # 中指
    (0, 13, 14, 15, 16),  # 无名指
    (0, 17, 18, 19, 20)  # 小指
]]


def update(frame):
    # 获取最新200ms肌电数据（示例用随机数据演示）
    emg_window = np.random.randn(200)  # 替换为实际数据采集

    # 预测关键点
    with torch.no_grad():
        pred_kp = model(torch.FloatTensor(emg_window).unsqueeze(0))[0].numpy()

    # 更新散点
    scatter.set_offsets(pred_kp)

    # 更新连线
    for line, indices in zip(lines, [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                                     (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]):
        x = [pred_kp[i][0] for i in indices]
        y = [pred_kp[i][1] for i in indices]
        line.set_data(x, y)

    return [scatter] + lines


# 运行动画
ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.xlim(0, 640)  # 假设原始图像宽度640px
plt.ylim(0, 480)  # 假设原始图像高度480px
plt.gca().invert_yaxis()  # 反转y轴匹配图像坐标系
plt.show()