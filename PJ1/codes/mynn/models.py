from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, reg_method=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    if reg_method == 'L2_Reg':
                        layer.weight_decay = True
                        l2_layer = L2Regularization(layer, reg_coeff=lambda_list[i]) 
                        self.layers.append(l2_layer)
                    else:
                        layer.weight_decay = True
                        layer.weight_decay_lambda = lambda_list[i]
                        self.layers.append(layer)
                else:
                    self.layers.append(layer)

                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        super().__init__()
        self.layers = [
            conv2D(in_channels=1, out_channels=8, kernel_size=3),   # 输出26x26
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),                     # 输出13x13
            conv2D(in_channels=8, out_channels=16, kernel_size=3),  # 输出11x11
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),                     # 输出5x5 
            Flatten(),                                              # 展平为16*5*5=400维
            Linear(in_dim=16*5*5, out_dim=64),            
            ReLU(),
            Linear(in_dim=64, out_dim=10)            
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape(-1, 1, 28, 28)
            
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        
        layer_idx = 0
        for param in param_list:
            if isinstance(param, dict):  
                current_layer = self.layers[layer_idx]
                while not hasattr(current_layer, 'params'): 
                    layer_idx += 1
                    current_layer = self.layers[layer_idx]
                
                current_layer.W = param['W']
                current_layer.b = param['b']
                current_layer.params = {'W': current_layer.W, 'b': current_layer.b}
                layer_idx += 1
        
    def save_model(self, save_path):
        """
        保存模型参数（参考Model_MLP实现）
        """
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'params'):  
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b']
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)