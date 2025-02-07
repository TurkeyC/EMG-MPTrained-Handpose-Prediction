import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO

# 加载YOLOv11模型
model = YOLO(r"YOLOv11_HandPose/YOLO_Models/Pose_Models/yolo11n-pose.pt")

# 图片文件夹路径
image_folder = r"YOLOv11_HandPose/Test_Dataset/Handpose_Notip_Gra"

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 处理每张图片
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    
    # YOLOv11进行姿态估计
    results = model(image)
    
    # 绘制检测结果
    for result in results:
        x1, y1, x2, y2, confidence, class_id = result
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_id} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 显示结果图片
    cv2.imshow('Hand Pose Estimation', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()