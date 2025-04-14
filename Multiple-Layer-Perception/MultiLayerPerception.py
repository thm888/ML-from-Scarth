from abc import ABC,abstractmethod
import numpy as np 
from typing import Optional

class Layer(ABC): #继承 ABC 使得不能直接实例化 Layer
    def __init__(self):
        """缓存向前传播的激活值，用于反向传播"""
        self.input = None

    @abstractmethod #使用abstractmethod装饰器使得必须重写方法
    def forward(self, x: np.ndarray) -> np.ndarray:
        """向前传播，输入 x, 返回该层的输出"""
        pass 

    @abstractmethod
    def backward(self, x:np.ndarray) -> np.ndarray:
        """反向传播，输入上层的梯度，返回该层的梯度，并计算参数梯度（如果有参数）"""
        pass 

    def parameters(self) -> list:
        """返回本层的可训练参数"""
        return []

    def gradients(self) -> list:
        """返回本层的梯度参数"""
        return []

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """重载括号运算符，方便调用output = Layer(input)""" 
        return self.forward(x)

class Linear(Layer):
	def __init__(self, input_dim:int, output_dim:int):
		super().__init__()
		#参数初始化
		self.weights = np.random.rand(output_dim, input_dim) * np.sqrt(2.0/input_dim)
		self.bias = np.zeros((output_dim,1))
		#梯度
		self.grad_weights = np.zeros_like(self.weights)
		self.grad_bias = np.zeros_like(self.bias)
		
	def forward(self, x:np.ndarray) -> np.ndarray: 
		self.input = x 
		return  np.dot(self.weights,x) + self.bias 
		
	def backward(self, grad_output:np.ndarray) -> np.ndarray:
		#计算本层梯度
		grad_input = np.dot(self.weights.T , grad_output)
		self.grad_weights = np.dot(grad_output, self.input.T)
		self.grad_bias =  np.sum(grad_output, axis = 1, keepdims=True)
		return grad_input
		
	def parameters(self) ->list:
		return [self.weights,self.bias]
		
	def gradients(self) -> list:
		return [self.grad_weights,self.grad_bias]
      
class Relu(Layer):
	def forward(self, x:np.ndarray) -> np.ndarray:
		self.input = x #缓存输入，用于激活
		return np.maximum(0,x)
		
	def backward(self, grad_output:np.ndarray) -> np.ndarray:
		mask = (self.input > 0).astype(float) #激活值＞0才传递梯度
		return grad_output * mask 
	
class Dropout(Layer):
	def __init__(self, p:float = 0.8):
		super().__init__()
		self.p = p #丢弃概率
		self.mask:Optional[np.ndarray] = None
		self.training = True #是否为训练模式
		
	def train(self):
		"""训练模式"""
		self.training = True
		
	def eval(self):
		"""推理模式"""
		self.traning = False
		
	def forward(self, x:np.ndarray, training = None) ->np.ndarray:
		if training is None:
			training = self.training
			
		if training:
			#保持期望不变
			self.mask = (np.random.rand(*x.shape) > self.p )/(1-self.p)
			return x * self.mask
			
		else:
			#推理时直接返回
			return x 
			
	def backward(self, grad_output: np.ndarray) -> np.ndarray:
		return grad_output * self.mask	

class MLP:
	def __init__(self,layers):
		self.layers = layers
		self.tarining_mode = True
		
	def train(self):
		"""训练模式"""
		self.training_mode = True
		for layer in self.layers:
			if isinstance(layer, Dropout):
				layer.train()
				
	def eval(self):
		"""推理模式"""
		self.training_mode = False
		for layer in self.layers:
			if isinstance(layer, Dropout):
				layer.eval()
	
	def forward(self, x):
		"""向前传播"""
		for layer in self.layers:
			if isinstance(layer, Dropout):
				x = layer.forward(x, training = self.training_mode)
			else:
				x = layer.forward(x)
		return x 
	
	def backward(self, grad):
		"""反向传播"""
		for layer in reversed(self.layers):
			grad = layer.backward(grad)
		return grad 
		
	def parameters(self):
		"""获取参数"""
		params = []
		for layer in self.layers:
			params.extend(layer.parameters())
		return params
	
	def zero_grads(self):
		"""梯度清零"""
		for layer in self.layers:
			if isinstance(layer, Linear):
				layer.grad_weights.fill(0)
				layer.grad_bias.fill(0)
	
	def grads(self):
		"""获取梯度"""
		grads = []
		for layer in self.layers:
			grads.extend(layer.gradients())
		return grads
	

class OutputSoftmax(Layer):
	def forward(self, x:np.ndarray)-> np.ndarray:
		x = x - np.max(x , axis = 0, keepdims= True)
		self.output = np.exp(x)/np.sum(np.exp(x), axis = 0 ,keepdims = True)
		return self.output 
		
	def backward(self, grad_output: np.ndarray) -> np.ndarray:
		return grad_output 
	
class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        """
        y_pred: 模型输出（Softmax 后的概率），形状 (num_classes, batch_size)
        y_true: 真实标签的 one-hot 编码，形状 (num_classes, batch_size)
        """
        batch_size = y_pred.shape[-1]
        #计算loss
        loss = -np.sum(np.log(y_pred + 1e-8) * y_true )/batch_size
        #存储梯度
        self.grad = (y_pred - y_true)/batch_size #梯度显式解
        return loss 

    def backward(self):
        return self.grad  #回传梯度到OutputSoftmax
	
class SGD:
	def __init__(self, learning_rate = 1e-2, momentum = 0.0):
		self.lr = learning_rate 
		self.momentum = momentum 
		self.velocities = {} #保存每个参数的动量（按 id(parm)索引）
	
	def step(self, layers:list):
		for layer in layers:
			params = layer.parameters()
			grads = layer.gradients()
			for i, (param,grad) in enumerate(zip(params,grads)):
				#为每个param 生成唯一标识符（用于跨 step 跟踪） 
				param_id = id(param)
				if param_id not in self.velocities:
					self.velocities[param_id] = np.zeros_like(param)
				#动量更新
				self.velocities[param_id] = self.momentum * self.velocities[param_id] - self.lr * grad
				param += self.velocities[param_id] 

class Adam:
	def __init__(self, learning_rate = 1e-4, beta1 = 0.90, beta2 = 0.999, epsilon = 1e-8):
		self.lr = learning_rate 
		self.beta1 = beta1 
		self.beta2 = beta2 
		self.epsilon = epsilon
		self.t = 0 
		self.m = {}
		self.v = {}
	
	def step(self, layers:list):
		self.t += 1 
		for layer in layers:
			params = layer.parameters()
			grads = layer.gradients()
			for i, (param, grad) in enumerate(zip(params,grads)):
				param_id = id(param)
				if param_id not in self.m:
					self.m[param_id] = np.zeros_like(param)
					self.v[param_id] = np.zeros_like(grad)
				#动量更新
				self.m[param_id] = self.beta1 * self.m[param_id] + ( 1 -self.beta1) * grad
				self.v[param_id] = self.beta2 * self.v[param_id] + ( 1 - self.beta2) * (grad**2)
				#修正动量
				m_hat = self.m[param_id]/(1 - self.beta1 ** self.t)
				v_hat = self.v[param_id]/(1 - self.beta2 ** self.t)
				#参数更新
				param_update = -self.lr * m_hat/(np.sqrt(v_hat)+self.epsilon)
				param += param_update 			 
				
if __name__ == '__main__':
	#模型初始化
	nnSequence = [
		Linear(784,256),
		Relu(),
		Dropout(p=0.5),
		Linear(256,128),
		Relu(),
		Dropout(p=0.3),
		Linear(128,10),
		OutputSoftmax()
	]
	
	mlp = MLP(nnSequence)
	optimazor = Adam()
	criterion = CrossEntropyLoss()
	#初始化测试参数
	x = np.random.randn(784,32)
	y_true = np.eye(10)[np.random.choice(10,32)].T
	#开始训练
	mlp.train()
	y_pred = mlp.forward(x)
	loss = criterion(y_pred, y_true)
	#反向传播
	grad = criterion.backward()
	mlp.backward(grad)
	#参数更新
	optimazor.step(mlp.layers)

	#进行推理
	mlp.eval()
	y_pred_eval = mlp.forward(x)
	print(y_pred_eval.argmax(axis=0))
