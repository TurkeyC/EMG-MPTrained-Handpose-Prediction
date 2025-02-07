import torch
import torch.nn as nn
import cv2
import numpy as np

# 定义模型结构
class HandKeypointsModel(nn.Module):
    def __init__(self, max_lines=6, max_points=5):
        super(HandKeypointsModel, self).__init__()
        self.max_lines = max_lines
        self.max_points = max_points
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),  # 根据输入图片大小调整
            nn.ReLU(),
            nn.Linear(512, max_lines * max_points * 2),  # 动态输出维度
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 加载模型权重
model = HandKeypointsModel(max_lines=6, max_points=5)
model.load_state_dict(torch.load(r"YOLOv11_HandPose/Model_Training/Model_Output/hand_keypoints_model02.pt"))
model.eval()  # 设置为评估模式

# 预处理输入图片
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    return image

# 可视化关键点
def visualize_keypoints(image_path, keypoints):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for x, y in keypoints:
        if x > 0 and y > 0:  # 过滤掉填充的 (0, 0) 点
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)
    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例：推理并可视化结果
image_path = "YOLOv11_HandPose/Model_Training/Hand_Keypoints_Dataset/Images_Data/train/AH30.jpg"
input_tensor = preprocess_image(image_path)

# 将输入数据移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
model = model.to(device)

# 运行推理
with torch.no_grad():
    outputs = model(input_tensor)

# 将输出转换为关键点坐标
keypoints = outputs.cpu().numpy().reshape(-1, 2)

# 可视化结果
visualize_keypoints(image_path, keypoints)