import matplotlib.pyplot as plt

# 邻居数量
neighbors = [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23]

# 对应的准确率
accuracies = [0.3358, 0.3386, 0.3414, 0.3429, 0.3417, 0.3407, 0.3405, 0.3402, 0.341, 0.339, 0.3398, 0.3375, 0.3349]

# 绘制图像
plt.plot(neighbors, accuracies, marker='o', linestyle='-')
plt.title('Accuracy vs Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
