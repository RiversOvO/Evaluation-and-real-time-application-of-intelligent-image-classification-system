from tensorflow import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# 加载 CIFAR-10 数据集并将其拆分为训练集和测试集
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 数据正则化（将像素值缩放到 [0, 1] 范围内）
x_train = x_train / 255.0
x_test = x_test / 255.0

# 将数据重塑为二维数组
x_train2 = x_train.reshape((len(x_train), -1))
x_test2 = x_test.reshape((len(x_test), -1))

# 创建随机森林分类器模型
model = RandomForestClassifier(n_estimators=500)

# 计时开始
start_time = time.time()

# 使用训练集训练模型
model.fit(x_train2, y_train)

# 使用模型对测试集进行预测
y_pred = model.predict(x_test2)

# 计时结束
end_time = time.time()

# 计算预测准确率
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)

# 打印分类报告
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
file_name = "RF500.txt"
with open(results_folder + file_name, "w") as file:
    file.write("Accuracy: " + str(accuracy) + "\n\n")
    file.write("Classification Report:\n" + report + "\n\n")
    file.write("Confusion Matrix:\n" + np.array2string(confusion) + "\n\n")
    file.write("Run Time: " + str(run_time) + " seconds")
