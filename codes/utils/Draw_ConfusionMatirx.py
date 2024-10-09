import matplotlib.pyplot as plt
import numpy as np

# 混淆矩阵数据
confusion_matrix = np.array([
    [910, 6, 29, 3, 9, 1, 2, 1, 23, 16],
    [9, 940, 1, 0, 0, 0, 1, 0, 7, 42],
    [11, 1, 904, 13, 28, 5, 27, 7, 2, 2],
    [11, 1, 67, 720, 40, 89, 50, 11, 4, 7],
    [6, 1, 46, 18, 868, 13, 23, 20, 3, 2],
    [1, 0, 45, 122, 34, 765, 14, 19, 0, 0],
    [6, 1, 22, 24, 10, 3, 930, 2, 1, 1],
    [12, 2, 26, 13, 36, 37, 2, 868, 2, 2],
    [34, 7, 11, 2, 2, 1, 1, 1, 931, 10],
    [6, 37, 1, 2, 0, 0, 1, 1, 12, 940]
])

# 标准化混淆矩阵
confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# 标签
labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# 绘制热图
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)

# 显示值
for i in range(confusion_matrix_normalized.shape[0]):
    for j in range(confusion_matrix_normalized.shape[1]):
        plt.text(j, i, format(confusion_matrix_normalized[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_normalized[i, j] > 0.5 else "black")

plt.title('Normalized Confusion Matrix Heatmap')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
