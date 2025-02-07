from ultralytics import YOLO
import cv2
import os

# 1. 加载YOLOv11姿势估计模型
def load_model(model_path=r"YOLOv11_HandPose/YOLO_Models/Pose_Models/yolo11x-pose.pt"):
    """
    加载YOLOv11姿势估计模型。
    :param model_path: 模型权重文件路径
    :return: 加载的模型
    """
    model = YOLO(model_path)  # 加载预训练的YOLOv11姿势估计模型
    return model

# 2. 对单张图片进行推理
def infer_image(model, image_path, save_dir="Annotation_Results"):
    """
    对单张图片进行推理并保存结果。
    :param model: 加载的YOLOv11模型
    :param image_path: 图片路径
    :param save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 对图片进行推理
    results = model(image_path)  # 推理

    # 可视化结果并保存
    for result in results:
        # 绘制关键点和姿态连线
        annotated_frame = result.plot()  # 自动绘制关键点和姿态连线

        # 保存结果图片
        output_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_frame)
        print(f"结果已保存到: {output_path}")

# 3. 对文件夹中的所有图片进行推理
def infer_folder(model, image_folder, save_dir="Annotation_Results"):
    """
    对文件夹中的所有图片进行推理。
    :param model: 加载的YOLOv11模型
    :param image_folder: 图片文件夹路径
    :param save_dir: 结果保存目录
    """
    # 遍历文件夹中的图片
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 跳过非图片文件

        # 对每张图片进行推理
        infer_image(model, image_path, save_dir)

# 主函数
def main():
    # 参数设置
    #model_path = r"YOLOv11_HandPose/YOLO_Models/Pose_Models/yolo11x-pose.pt"  # YOLOv11姿势估计模型权重路径
    model_path = r"YOLOv11_HandPose/Model_Training/YOLO11n-Pose-Hands-Official_Demo/runs/pose/train/weights/best.pt"
    image_folder = r"YOLOv11_HandPose/Test_Dataset/Handpose_Notip_Gra"  # 图片文件夹路径
    save_dir = r"YOLOv11_HandPose/Test_Dataset/Annotation_Results"  # 结果保存目录

    # 加载模型
    model = load_model(model_path)

    # 对文件夹中的所有图片进行推理
    infer_folder(model, image_folder, save_dir)

if __name__ == "__main__":
    main()