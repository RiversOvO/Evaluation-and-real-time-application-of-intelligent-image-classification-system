from tensorflow import keras  # TensorFlow 中的 Keras 模块
import matplotlib.pyplot as plt  # 用于绘图的 Matplotlib 库
import numpy as np  # 处理数组的 NumPy 库
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # 评估指标
from sklearn.neighbors import KNeighborsClassifier
import time

# 加载 CIFAR-10 数据集并将其拆分为训练集和测试集
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# CIFAR-10 数据集中的类别标签
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 数据正则化（将像素值缩放到 [0, 1] 范围内）
x_train = x_train / 255.0
x_test = x_test / 255.0

# 获取训练集 x_train 的形状信息，包括样本数量、图像宽度、图像高度和通道数
nsamples, nx, ny, nrgb = x_train.shape

# 将 x_train 数据的形状重新调整为二维数组，以符合 scikit-learn 模型的输入要求
# 第一维表示样本数量，第二维表示每个样本的特征数量
x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))

# 获取测试集 x_test 的形状信息，包括样本数量、图像宽度、图像高度和通道数
nsamples, nx, ny, nrgb = x_test.shape

# 将 x_test 数据的形状重新调整为二维数组，以符合 scikit-learn 模型的输入要求
# 第一维表示样本数量，第二维表示每个样本的特征数量
x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

# 打印调整后的训练集和测试集的形状信息
print(x_train2.shape, x_test2.shape)

# 创建K最近邻分类器对象，指定邻居数量(可以修改)
knn = KNeighborsClassifier(n_neighbors=23)

# 计时开始
start_time = time.time()

# 使用训练数据拟合（训练）K最近邻分类器
knn.fit(x_train2, y_train)

# 对测试数据进行预测
y_pred = knn.predict(x_test2)

# 计时结束
end_time = time.time()


# 计算预测准确率
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)

# 打印分类报告，包括精确度、召回率和 F1 值
report = classification_report(y_pred, y_test)
print("Classification Report:\n", report)

# 打印混淆矩阵
confusion = confusion_matrix(y_pred, y_test)
print("Confusion Matrix:\n", confusion)

# 计算运行时间
run_time = end_time - start_time
print("Run Time:", run_time, "seconds")

# 将结果保存至文件夹 results 中
results_folder = "results/"
file_name = "KNN23.txt"
with open(results_folder + file_name, "w") as file:
    file.write("Accuracy: " + str(accuracy) + "\n\n")
    file.write("Classification Report:\n" + report + "\n\n")
    file.write("Confusion Matrix:\n" + np.array2string(confusion) + "\n\n")
    file.write("Run Time: " + str(run_time) + " seconds")