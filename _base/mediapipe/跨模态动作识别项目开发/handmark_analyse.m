data = readtable('hand_landmarks_n250205temp.csv'); 

frame_id = data.frame_id;  % 时间帧
x_coords = data{:, 2:2:end};  % 提取所有 x 坐标（偶数列）
y_coords = data{:, 3:2:end};  % 提取所有 y 坐标（奇数列）


%%%%%%%%%%%%%%%%%% 创建视频文件
video = VideoWriter('hand_movement.avi');
open(video);
%%%%%%%%%%%%%%%%%%


figure;
hold on;
axis equal;
xlim([min(x_coords(:)) max(x_coords(:))]);  % 设置 X 轴范围
ylim([min(y_coords(:)) max(y_coords(:))]);  % 设置 Y 轴范围
xlabel('X Coordinate');
ylabel('Y Coordinate');
title('Hand Keypoints Movement Over Time');

for frame = 1:length(frame_id)
    plot(x_coords(frame, :), y_coords(frame, :), 'o-', 'LineWidth', 2);  % 绘制关键点
    
    title(['Hand Keypoints Movement Over Time - Frame ', num2str(frame)]); % 更新标题，显示当前帧号
    
    drawnow;  % 更新图形
%%%%%%%%%%%%
    % pause(0.01);  % 控制动画速度
%%%%%%%%%%%%
    frame_img = getframe(gcf);  % 捕获当前帧
    writeVideo(video, frame_img);  % 写入视频
%%%%%%%%%%%%
    if frame < length(frame_id)
        cla;  % 清除当前帧，准备绘制下一帧
    end
end
hold off;


%%%%%%%%%%%%%%
close(video);  % 关闭视频文件