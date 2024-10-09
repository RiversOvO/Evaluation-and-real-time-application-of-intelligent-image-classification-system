import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np

# 设备设置为 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 VGG 网络
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)  # CIFAR-10 数据集有 10 个类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 加载训练好的模型
model = VGG().to(device)
model.load_state_dict(torch.load('vgg_cifar10.pth'))  # 假设模型已经保存为 vgg_cifar10.pth
model.eval()

# 定义图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1] 范围
])

# CIFAR-10 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图像
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理图像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 模型预测
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    # 在图像上显示预测结果
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示图像
    cv2.imshow('Real-time Image Classification', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()