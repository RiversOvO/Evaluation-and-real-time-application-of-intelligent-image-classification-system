from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 将分类报告信息存储在字典中
report = {
    "precision": [0.90, 0.94, 0.78, 0.79, 0.85, 0.84, 0.88, 0.93, 0.95, 0.92],
    "recall": [0.91, 0.94, 0.90, 0.72, 0.87, 0.77, 0.93, 0.87, 0.93, 0.94],
    "f1-score": [0.91, 0.94, 0.84, 0.75, 0.86, 0.80, 0.91, 0.90, 0.94, 0.93],
    "class": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}

# 绘制precision、recall、f1-score
plt.figure(figsize=(10, 6))
plt.plot(report["class"], report["precision"], label='Precision', marker='o')
plt.plot(report["class"], report["recall"], label='Recall', marker='o')
plt.plot(report["class"], report["f1-score"], label='F1-score', marker='o')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision, Recall, F1-score by Class')
plt.legend()

# 标出具体数值
for i in range(len(report["class"])):
    plt.text(report["class"][i], report["precision"][i], str(report["precision"][i]), ha='right')
    plt.text(report["class"][i], report["recall"][i], str(report["recall"][i]), ha='left')
    plt.text(report["class"][i], report["f1-score"][i], str(report["f1-score"][i]), ha='center')

plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 绘制weighted avg
weighted_avg = 0.88, 0.88, 0.88
plt.figure(figsize=(6, 4))
bars = plt.bar(['Precision', 'Recall', 'F1-score'], weighted_avg, color=['blue', 'green', 'orange'])
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Weighted Avg: Precision, Recall, F1-score')
plt.ylim(0, 1)

# 标出具体数值
for bar, value in zip(bars, weighted_avg):
    plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom')

plt.show()
