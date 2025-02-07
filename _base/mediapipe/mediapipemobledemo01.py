import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# 配置参数
screen_w, screen_h = pyautogui.size()
DISTANCE_THRESHOLD = 0.18  # 触发点击的距离阈值

# 初始化Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 摄像头设置
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 状态变量
mouse_down = False

def get_palm_center(landmarks):
    """计算并返回手掌中心坐标（带可视化）"""
    # 通过手腕和中指根部计算中心点
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    palm_x = (wrist.x + middle_mcp.x) / 2
    palm_y = (wrist.y + middle_mcp.y) / 2
    return palm_x, palm_y

def check_click(landmarks):
    """检测四指指尖是否靠近掌心"""
    palm_x, palm_y = get_palm_center(landmarks)
    
    # 计算手部参考尺寸（食指到小指根部宽度）
    hand_width = abs(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x -
                     landmarks[mp_hands.HandLandmark.PINKY_MCP].x)
    
    # 检测四指指尖（排除拇指）
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    for tip in finger_tips:
        tip_x = landmarks[tip].x
        tip_y = landmarks[tip].y
        distance = ((tip_x - palm_x)**2 + (tip_y - palm_y)**2)**0.5 / hand_width
        if distance < DISTANCE_THRESHOLD:
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    debug_image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # 绘制手部关键点（排除拇指）
        connections = mp_hands.HAND_CONNECTIONS
        filtered_connections = [conn for conn in connections 
                               if conn[0] not in [0,1,2,3,4] and 
                                  conn[1] not in [0,1,2,3,4]]
        mp.solutions.drawing_utils.draw_landmarks(
            debug_image, hand_landmarks, filtered_connections,
            mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=3),
            mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=3)
        )

        # ========== 新增：绘制掌心中心 ==========
        palm_x, palm_y = get_palm_center(landmarks)
        cx, cy = int(palm_x * 640), int(palm_y * 480)
        cv2.circle(debug_image, (cx, cy), 12, (0,165,255), -1)  # 橙色中心点
        # ======================================

        # 鼠标移动控制
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_tip.x * screen_w)
        y = int(index_tip.y * screen_h)
        pyautogui.moveTo(x, y)

        # 点击检测
        if check_click(landmarks):
            if not mouse_down:
                pyautogui.mouseDown()
                mouse_down = True
                cv2.putText(debug_image, "CLICKING", (200,100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        else:
            if mouse_down:
                pyautogui.mouseUp()
                mouse_down = False

    # 状态显示
    cv2.putText(debug_image, f"CLICK: {'ON' if mouse_down else 'OFF'}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('Palm Center Mouse', debug_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()