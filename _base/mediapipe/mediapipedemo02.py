# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
from concurrent.futures import ThreadPoolExecutor

# 配置区（新手可调参数）=================================================
USE_CAMERA = True          # 是否使用摄像头（False则读取视频文件）
VIDEO_PATH = "test.mp4"    # 视频文件路径
HISTORY_LEN = 5            # 滑动平均窗口大小（3-7）
MIN_DETECTION_CONF = 0.8   # 手部检测置信度阈值（0-1）
SHOW_RAW_POINTS = True     # 是否显示原始检测点

# 初始化线程池=========================================================
executor = ThreadPoolExecutor(max_workers=4)  # 线程池（加速卡尔曼滤波）

# 初始化OpenCV=========================================================
#cv2.cuda.setDevice(0)  # 启用GPU加速（需NVIDIA GPU）

# 初始化MediaPipe手部模型===============================================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=0.6)

# 初始化卡尔曼滤波器组（每个关键点一个滤波器）============================
class KalmanFilterWrapper:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # 状态转移矩阵（假设匀速运动）
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])
        # 观测矩阵
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]])
        # 过程噪声协方差
        self.kf.Q = np.eye(4) * 0.01
        # 观测噪声协方差
        self.kf.R = np.array([[5,0],[0,5]])  

kf_array = [KalmanFilterWrapper() for _ in range(21)]  # 21个关键点

# 初始化滑动平均容器====================================================
history = deque(maxlen=HISTORY_LEN)

# 视频流处理主循环======================================================
cap = cv2.VideoCapture(0 if USE_CAMERA else VIDEO_PATH)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # 预处理
    image = cv2.flip(image, 1)  # 镜像翻转
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]  # 获取画面尺寸
    
    # 手部检测
    results = mp_hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        current_landmarks = []
        
        # 处理每个关键点
        for idx, lm in enumerate(hand_landmarks.landmark):
            # 卡尔曼滤波
            kf = kf_array[idx]
            kf.kf.predict()
            kf.kf.update(np.array([lm.x, lm.y]))
            
            # 获取平滑后坐标
            smoothed_x = kf.kf.x[0]
            smoothed_y = kf.kf.x[1]
            current_landmarks.append([smoothed_x, smoothed_y])
            
            # 绘制原始点（绿色）
            if SHOW_RAW_POINTS:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
        
        # 滑动平均滤波
        history.append(current_landmarks)
        avg_landmarks = np.mean(history, axis=0) if history else current_landmarks
        
        # 绘制最终平滑点（红色）
        for point in avg_landmarks:
            cx, cy = int(point[0] * w), int(point[1] * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    
    # 显示画面
    cv2.imshow('Stabilized Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()