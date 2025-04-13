import numpy as np 
from logistic_REG import LogisticRegression
from sklearn.model_selection import train_test_split

# 读取CSV文件

data = np.loadtxt('./LearnMachineLearning/LogisticRegression/lr_dataset.csv',delimiter=',')
# 分割为特征X和标签y
X = data[:, :-1]  # 前两列是特征
y = data[:, -1]   # 最后一列是标签
y = y.astype(int)

# 划分数据集（一般设置 20%-30% 作为测试集）
X_train, X_test, y_train, y_t = train_test_split(
    X, 
    y, 
    test_size=0.2,        # 测试集比例
    random_state=42,      # 随机种子（确保可复现性）
    stratify=y            # 保持类别分布（可选，推荐用于分类任务）
)

model = LogisticRegression(n_classes=2,learning_rate=1e-2,max_epoches=50,tol=1e-6, verbose=True, batch_size=10, print_every=10)
model.fit(X_train,y_train)
probabilities = model.predict(X_test) #(N,C)
y_pred = probabilities.argmax(axis=1) #(N,1)
acc = np.mean(y_pred == y_t)
print(acc)
print(model.check_parameter())

