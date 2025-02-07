import onnx
import torch
from onnx2pytorch import ConvertModel

# 加载 ONNX 模型
onnx_model = onnx.load(r"YOLOv11_HandPose/Format_Conversion/onnx2pytorch/hand_keypoints_model.onnx")

# 转换为 PyTorch 模型
model_torch = ConvertModel(onnx_model)

# 保存为 PyTorch 的 .pt 文件
torch.save(model_torch.state_dict(), "hand_keypoints_model.pt")