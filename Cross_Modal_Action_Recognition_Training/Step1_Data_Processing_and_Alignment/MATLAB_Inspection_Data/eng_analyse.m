% 读取 CSV 文件
data = readmatrix('emg_data_n250205temp.csv');

% 提取 X 和 Y 数据
x = data(:, 1);
y = data(:, 2);

% 绘制图像
plot(x, y);
xlabel('time_ms');
ylabel('value');
title('emg_analyse');
grid on;

% 指定坐标轴范围（可选）
xlim([0, 5400000]);  % 设置 X 轴范围
ylim([200, 400]);  % 设置 Y 轴范围