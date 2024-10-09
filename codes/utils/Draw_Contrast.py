import matplotlib.pyplot as plt
import pandas as pd

# 创建数据框
data = {
    'Model': ['KNN', 'Random Forest', 'Naive Bayes', 'VGG'],
    'Time (s)': [7.049453973770142, 162.93877053260803, 2.77211332321167, 262.5114161968231],
    'Accuracy': [0.3429, 0.4634, 0.2976, 0.8776],
    'Weighted Precision': [0.47, 0.47, 0.39, 0.88],
    'Weighted Recall': [0.34, 0.46, 0.30, 0.88],
    'Weighted F1': [0.36, 0.47, 0.32, 0.88]
}
df = pd.DataFrame(data)

# 打印表格
print(df)

# 绘制图表
plt.figure(figsize=(16, 8))

# 时间
plt.subplot(2, 3, 1)
plt.bar(df['Model'], df['Time (s)'], color='skyblue')
plt.title('Time (s)')
plt.ylabel('Seconds')
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f')) # 设置y轴格式
for i, v in enumerate(df['Time (s)']):
    plt.text(i, v, '{:.4f}'.format(v), ha='center', va='bottom')

# 准确率
plt.subplot(2, 3, 2)
plt.bar(df['Model'], df['Accuracy'], color='salmon')
plt.title('Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
for i, v in enumerate(df['Accuracy']):
    plt.text(i, v, '{:.4f}'.format(v), ha='center', va='bottom')

# 加权平均精确率
plt.subplot(2, 3, 3)
plt.bar(df['Model'], df['Weighted Precision'], color='lightgreen')
plt.title('Weighted Precision')
plt.ylim(0, 1)
plt.ylabel('Precision')
for i, v in enumerate(df['Weighted Precision']):
    plt.text(i, v, '{:.4f}'.format(v), ha='center', va='bottom')

# 加权平均召回率
plt.subplot(2, 3, 4)
plt.bar(df['Model'], df['Weighted Recall'], color='gold')
plt.title('Weighted Recall')
plt.ylim(0, 1)
plt.ylabel('Recall')
for i, v in enumerate(df['Weighted Recall']):
    plt.text(i, v, '{:.4f}'.format(v), ha='center', va='bottom')

# 加权平均 F1 值
plt.subplot(2, 3, 5)
plt.bar(df['Model'], df['Weighted F1'], color='lightblue')
plt.title('Weighted F1')
plt.ylim(0, 1)
plt.ylabel('F1 Score')
for i, v in enumerate(df['Weighted F1']):
    plt.text(i, v, '{:.4f}'.format(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()
