# 导入工具包（不用理解原理，直接复制）
import cv2
import mediapipe as mp

# 初始化MediaPipe手部模型
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,       # 视频模式
    max_num_hands=1,               # 只检测一只手
    min_detection_confidence=0.7)  # 置信度阈值

# 读取视频文件（将'test.mp4'换成你的视频名）
#cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取一帧画面
    success, image = cap.read()
    if not success:
        break
    
    # 将画面转换为RGB格式（MediaPipe需要）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测
    results = mp_hands.process(image_rgb)
    
    # 如果有检测到手
    if results.multi_hand_landmarks:
        # 获取第一个手的21个关键点坐标
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 在画面上绘制关键点和连接线
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    # 显示处理后的画面
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()