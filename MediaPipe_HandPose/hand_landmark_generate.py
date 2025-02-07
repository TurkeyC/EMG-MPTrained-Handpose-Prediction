import cv2
import mediapipe as mp
import pandas as pd

# 初始化MediaPipe Hands模型
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,       # 视频流模式
    max_num_hands=1,               # 只检测单手
    min_detection_confidence=0.7,  # 检测置信度阈值
    min_tracking_confidence=0.5    # 跟踪置信度阈值
)

# 视频处理函数
def process_video(input_path, output_csv):
    cap = cv2.VideoCapture(input_path)
    all_landmarks = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 转换为RGB格式
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)
        
        # 提取关键点
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                # 保存归一化坐标(x,y,z) 和可见性
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            all_landmarks.append(landmarks)
        else:
            all_landmarks.append([-1]*21*4)  # 未检测到手时填充-1
    
    # 保存为CSV
    columns = []
    for i in range(21):
        columns += [f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}']
    df = pd.DataFrame(all_landmarks, columns=columns)
    df.to_csv(output_csv, index_label='frame_id')
    cap.release()

# 使用示例
if __name__ == "__main__":
    process_video(
        #input_path='input_video.mp4',
        input_path='E:\\User_Stuff\\Documents\\Code_Typing\\Python_3.12_Code_Workspace\\SmallProject\\YOLO_handpose\\mediapipe\\sEMGproject_720x720.mp4',
        output_csv='hand_landmarks.csv'
    )