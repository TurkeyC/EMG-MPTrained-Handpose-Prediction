import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0  # 降低模型复杂度以提升速度
)

def process_video(input_path, output_csv):
    cap = cv2.VideoCapture(input_path)
    all_landmarks = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 获取原始图像尺寸
        h, w = frame.shape[:2]
        
        # 转换为RGB格式并保留尺寸信息
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 提升处理速度
        
        results = mp_hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                # 转换为绝对坐标（基于原始图像尺寸）
                abs_x = int(landmark.x * w)
                abs_y = int(landmark.y * h)
                landmarks.extend([abs_x, abs_y])
            all_landmarks.append(landmarks)
        else:
            all_landmarks.append([-1]*21*2)  # 仅保留x,y坐标

        # 在循环内添加以下代码
        debug_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow('Debug', debug_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 保存为CSV
    columns = []
    for i in range(21):
        columns += [f'x_{i}', f'y_{i}']
    df = pd.DataFrame(all_landmarks, columns=columns)
    df.to_csv(output_csv, index_label='frame_id')
    cap.release()

if __name__ == "__main__":
    process_video('E:\\User_Stuff\\Documents\\Code_Typing\\Python_3.12_Code_Workspace\\SmallProject\\YOLO_handpose\\mediapipe\\sEMGproject_480x480.mp4', 'hand_landmarks_n250205.csv')