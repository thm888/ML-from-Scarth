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

    for epoch in range(MAX_EPOCHES):
        #打乱数据排列
        indices = np.random.permutation(total_samples)
        X_shuffled = X_train[:,indices]
        y_shuffled = y_train[:,indices]
        step = 0
        #开始训练一个EPOCH
        for idx in range(0, total_samples, BATCH_SIZE):
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

    mlp.eval()
    acc = np.mean(mlp.forward(X_test).argmax(axis=0) == y_test)
    print("分类准确率:",acc)
    print('y_true',y_test)
    print(mlp.parameters())
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

main('xor_dataset.csv',1000,128) 
