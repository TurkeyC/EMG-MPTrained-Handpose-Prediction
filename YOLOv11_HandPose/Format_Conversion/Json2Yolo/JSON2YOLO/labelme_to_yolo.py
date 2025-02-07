# import os
# import json
# import numpy as np

# def convert_labelme_to_yolo(json_file, output_dir):
#     # 读取LabelMe的JSON文件
#     with open(json_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # 获取图片尺寸
#     image_width = data["imageWidth"]
#     image_height = data["imageHeight"]

#     # 创建输出文件路径
#     image_name = os.path.splitext(data["imagePath"])[0]
#     output_file = os.path.join(output_dir, f"{image_name}.txt")

#     # 打开输出文件
#     with open(output_file, "w", encoding="utf-8") as f:
#         # 遍历每个标注
#         for shape in data["shapes"]:
#             if shape["shape_type"] != "point":
#                 continue  # 只处理关键点标注

#             # 获取关键点坐标
#             points = np.array(shape["points"])
#             x, y = points[0]

#             # 归一化坐标
#             x_norm = x / image_width
#             y_norm = y / image_height

#             # 写入YOLO格式
#             f.write(f"0 {x_norm:.6f} {y_norm:.6f}\n")

# def batch_convert(json_dir, output_dir):
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)

#     # 遍历JSON目录中的所有文件
#     for json_file in os.listdir(json_dir):
#         if not json_file.endswith(".json"):
#             continue

#         # 转换单个JSON文件
#         json_path = os.path.join(json_dir, json_file)
#         convert_labelme_to_yolo(json_path, output_dir)

# if __name__ == "__main__":
#     # 设置输入和输出目录
#     json_dir = "YOLOv11_HandPose/Format_Conversion/Json2Yolo/JSON2YOLO/Pre_Data" # LabelMe JSON文件目录
#     output_dir = "YOLOv11_HandPose/Format_Conversion/Json2Yolo/JSON2YOLO/Yolo_Labels"  # YOLO格式标签输出目录

#     # 批量转换
#     batch_convert(json_dir, output_dir)
#     print(f"转换完成,YOLO标签已保存到: {output_dir}")


import os
import json
import numpy as np

def convert_labelme_to_yolo(json_file, output_dir):
    # 读取LabelMe的JSON文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取图片尺寸
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    # 创建输出文件路径
    image_name = os.path.splitext(data["imagePath"])[0]
    output_file = os.path.join(output_dir, f"{image_name}.txt")

    # 打开输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        # 遍历每个标注
        for shape in data["shapes"]:
            if shape["shape_type"] not in ["polygon", "linestrip"]:
                continue  # 只处理折线或多边形标注

            # 获取折线或多边形的点
            points = np.array(shape["points"])

            # 归一化坐标
            normalized_points = []
            for point in points:
                x, y = point
                x_norm = x / image_width
                y_norm = y / image_height
                normalized_points.extend([x_norm, y_norm])

            # 写入YOLO格式
            line = "0 " + " ".join(f"{coord:.6f}" for coord in normalized_points) + "\n"
            f.write(line)

def batch_convert(json_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历JSON目录中的所有文件
    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        # 转换单个JSON文件
        json_path = os.path.join(json_dir, json_file)
        convert_labelme_to_yolo(json_path, output_dir)

if __name__ == "__main__":
    # 设置输入和输出目录
    json_dir = r"YOLOv11_HandPose/Format_Conversion/Json2Yolo/JSON2YOLO/Pre_Data"  # LabelMe JSON文件目录
    output_dir = r"YOLOv11_HandPose/Format_Conversion/Json2Yolo/JSON2YOLO/Yolo_Labels"  # YOLO格式标签输出目录

    # 批量转换
    batch_convert(json_dir, output_dir)
    print(f"转换完成！YOLO标签已保存到：{output_dir}")