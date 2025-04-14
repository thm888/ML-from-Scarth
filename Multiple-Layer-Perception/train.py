from MultiLayerPerception import MLP,Linear,Relu,Dropout,OutputSoftmax,CrossEntropyLoss,Adam,SGD
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt 

#数据加载
def load_data(url:str, num_classes:int = 1):
    data = np.loadtxt(url,delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    y_train = np.eye(num_classes)[y_train]

    return X_train.T, X_test.T, y_train.T, y_test.T #(features, n_samples)

def main(url, MAX_EPOCHES=10,BATCH_SIZE=10):
    X_train, X_test, y_train, y_test = load_data(url = url,num_classes=2) #(features, n_samples)

    # 数据归一化
    X_train = X_train.T  # 转为 (n_samples, features)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_train = X_train.T  # 转回 (features, n_samples)

    X_test = X_test.T
    X_test = (X_test - mean) / std
    X_test = X_test.T

    #模型加载
    nnSequence = [
        Linear(2,4),
        Relu(),
        Linear(4,2),
        OutputSoftmax()
    ]

    mlp = MLP(nnSequence)
    mlp.train() 
    optimizor = SGD(learning_rate=1e-1)
    criterion = CrossEntropyLoss()
    total_samples = X_train.shape[-1]
    print('总样本数量',total_samples)
    loss_history = []
    acc_history = []

    for epoch in range(MAX_EPOCHES):
        #打乱数据排列
        indices = np.random.permutation(total_samples)
        X_shuffled = X_train[:,indices]
        y_shuffled = y_train[:,indices]
        step = 0
        #开始训练一个EPOCH
        for idx in range(0, total_samples, BATCH_SIZE):
            mlp.train()
            mlp.zero_grads() #梯度清零
            step += 1 
            X_batch = X_shuffled[:,idx: idx+BATCH_SIZE]
            y_batch = y_shuffled[:,idx: idx+BATCH_SIZE]
            #向前传递
            y_pred = mlp.forward(X_batch) 
            loss = criterion(y_pred,y_batch)
            loss_history.append(loss)
            print(f"epoch{epoch}/step{step} loss:",loss)
            #反向传播
            grad = criterion.backward()
            mlp.backward(grad)
            #参数更新
            optimizor.step(mlp.layers)
            #推理acc
            mlp.eval()
            acc = np.mean(mlp.forward(X_test).argmax(axis=0) == y_test)
            acc_history.append(acc)

    print(mlp.parameters())
    mlp.eval()
    acc = np.mean(mlp.forward(X_test).argmax(axis=0) == y_test)
    print('最后一次分类准确度:',acc)
    # 创建双纵坐标轴
    ax1 = plt.gca()  # 获取当前轴
    ax2 = ax1.twinx()

    # 绘制损失曲线（左轴）
    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_history, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 绘制准确率曲线（右轴）
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_history, color=color, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例和标题
    plt.title('Training Loss and Test Accuracy')
    fig = plt.gcf()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))  # 调整图例位置
    plt.grid(True)
    plt.show()

main('xor_dataset.csv',1000,256) 
