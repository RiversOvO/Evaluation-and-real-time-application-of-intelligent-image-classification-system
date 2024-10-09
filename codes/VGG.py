import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# 设备设置为 GPU，如果可用的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 CIFAR-10 数据集并将其转换为 PyTorch 的 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1] 范围
])

train_dataset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform)

# 创建 DataLoader 用于批量加载数据
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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


# 实例化 VGG 模型并将其移动到设备
model = VGG().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# 测试模型
@torch.no_grad()
def test_model(model, test_loader):
    model.eval()
    all_predictions = []
    true_labels = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # 计时结束
    end_time = time.time()

    accuracy = accuracy_score(true_labels, all_predictions)
    report = classification_report(true_labels, all_predictions)
    confusion = confusion_matrix(true_labels, all_predictions)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", confusion)

    # 计算运行时间
    run_time = end_time - start_time
    print("Run Time:", run_time, "seconds")

    # 保存结果到文件
    results_folder = "results/"
    file_name = "VGG.txt"
    with open(results_folder + file_name, "w") as file:
        file.write("Accuracy: " + str(accuracy) + "\n\n")
        file.write("Classification Report:\n" + report + "\n\n")
        file.write("Confusion Matrix:\n" + np.array2string(confusion) + "\n\n")
        file.write("Run Time: " + str(run_time) + " seconds")


# 训练和评估模型
num_epochs = 10

# 计时开始
start_time = time.time()

train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
test_model(model, test_loader)
