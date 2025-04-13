import numpy as np 

class LogisticRegression:
    def __init__(self, n_classes, 
        learning_rate = 1e-4, 
        max_epoches = 10, 
        tol = 1e-9, 
        random_state = 0, 
        verbose = False,
        batch_size = 1,
        print_every = 1,
        ):
        #超参数
        self.learning_rate = learning_rate
        self.max_epoches = max_epoches
        self.tol = tol 
        self.random_state = random_state
        self.verbose = verbose
        self.print_every = print_every
        self.batch_size = batch_size
        self.n_classes = n_classes 

    def _init_parameters(self, X):
        """
        输入 X;(1,n_features), 初始化模型参数
        """
        #全局参数
        rng = np.random.default_rng(seed=self.random_state)  # 局部种子

        self.n_features = X.shape[-1]
        self.W = rng.standard_normal(size=(self.n_features, self.n_classes))
        self.b = np.zeros(shape = (1, self.n_classes))
        self.W_gradient = np.zeros_like(self.W)
        self.b_gradient = np.zeros_like(self.b)
        self.loss_history = []

    def fit(self, X, y):
        """
        输入 X (sample_nums, n_features)，y (sample_nums), 训练模型
        """
        #初始化参数

        self._init_parameters(X[0])
        n_samples = X.shape[0]
        flag = False
        print('开始fit')
        for epoch in range(self.max_epoches):
            print(f"开始 epoch:{epoch}")
            #提前终止fit
            if flag:
                print("模型已经收敛，提前终止训练")
                break 

            #打乱samples
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            cnt = 0                
            for step in range(0, n_samples, self.batch_size):
                #去除 batch_size
                X_batch = X_shuffled[step:step+self.batch_size]
                y_batch = y_shuffled[step:step+self.batch_size]
                #向前传播
                print(f"开始向前传播 epoch:{epoch}/step:{cnt}")
                P, loss = self._forward(X_batch,y_batch,False)
                if np.abs(loss) <= self.tol:
                    flag = True
                    break 
                #打印loss
                self.loss_history.append(loss)
                if self.verbose:
                    if cnt % self.print_every == 0:
                        self._loss_print(cnt, epoch)
                #反向传播
                print(f"开始反向传播epoch:{epoch}/step:{cnt}")
                self._backward(X_batch, y_batch, P)
                #参数更新
                print(f"开始更新参数epoch:{epoch}/step:{cnt}")
                self._step()
                cnt += 1
            
    def _forward(self, X, y_true = None, if_predict = False):
        """
        向前传播过程
        输入 X (batch_size, n_features), y_true (batch_size),
            if_predict = False：是否在进行预测
        输出 P (batch_size, n_classes)，loss: 标量
        """

        if not if_predict and y_true is None:
            raise ValueError("训练模式下必须传入y_true")

        Z = X @ self.W + self.b #(B,C)
        P = self._softmax(Z, dim=-1) #(B,C)
        
        #预测模式
        if if_predict:
            return P
            
        #训练模式
        loss = self._compute_loss(P, y_true)
        return P,loss 

    def _softmax(self, z, dim = -1):
        """
        Softmax函数
        输入 z (batch_size, n_classes)，dim：进行 sum 的维度
        输出 z (barch_size, n_classes)
        """
        z = z - np.max(z, axis = dim, keepdims= True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z , axis = dim, keepdims = True)
        
    def _compute_loss(self, P, y_true):
        """
        计算损失
        输入 P（batch_size, n_classes), y_true (batch_size)
        输入 loss: float 标量
        """
        loss = 	np.mean(np.log(P[np.arange(0, self.batch_size), y_true]))
        return loss

    def _loss_print(self, step, epoch):
        """Print Loss"""
        print(f"Epoch {epoch}, Step {step}: Loss = {self.loss_history[-1]:.4f}")
    
    def _backward(self, X, y_true, P):
        """
        反向传播
        输入:X（batch_size, n_features）, y_true (batch_size), 
            P (batch_size, n_classes),
        输出：
            None
        """
        n_samples = len(y_true)
        one_hot = np.eye(self.n_classes)[y_true] #(B,C)
        back_loss = P - one_hot #(B,C)
        self.W_gradient = (X.T @ back_loss) / n_samples #(n_features, C)
        self.b_gradient = np.mean(back_loss, axis = 0) #(C,)

    def _step(self):
        """
        更新参数        
        Input ： None 
        Output ： None
        """
        #参数更新
        self.W -= self.learning_rate * self.W_gradient 
        self.b -= self.learning_rate * self.b_gradient
        #清空梯度
        self.W_gradient.fill(0)
        self.b_gradient.fill(0)
    
    def predict(self, X):
        return self._forward(X,if_predict=True) 

    def check_parameter(self):
        print("Parameter W:\n", self.W)
        print("Parameter b:\n", self.b)

    

