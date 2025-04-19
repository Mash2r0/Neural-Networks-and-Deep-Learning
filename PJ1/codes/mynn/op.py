from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(0, 1, size=(in_dim, out_dim))
        self.b = initialize_method(0, 1, size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        dW = np.dot(self.input.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)
        input_grad = np.dot(grad, self.W.T)
        if self.weight_decay:
            dW += 2 * self.weight_decay_lambda * self.W
        self.grads['W'] = dW
        self.grads['b'] = db
        return input_grad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 0
        
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.input = None
        
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch, in_channels, H, W = X.shape
        k = self.kernel_size

        H_out = (H - k) // self.stride + 1
        W_out = (W - k) // self.stride + 1
        
        output = np.zeros((batch, self.out_channels, H_out, W_out))
        
        for i in range(batch):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + k
                        w_start = w * self.stride
                        w_end = w_start + k

                        region = X[i, :, h_start:h_end, w_start:w_end]
                        output[i, c_out, h, w] = np.sum(region * self.W[c_out]) + self.b[0, c_out, 0, 0]
        
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        batch, in_channels, H_in, W_in = X.shape
        k = self.kernel_size
        
        dX = np.zeros_like(X)
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        
        for i in range(batch):
            for c_out in range(self.out_channels):
                for h in range(grads.shape[2]):
                    for w in range(grads.shape[3]):
                        h_start = h * self.stride
                        h_end = h_start + k
                        w_start = w * self.stride
                        w_end = w_start + k
                        
                        region = X[i, :, h_start:h_end, w_start:w_end]
                        self.grads['W'][c_out] += grads[i, c_out, h, w] * region
                        
                        dX[i, :, h_start:h_end, w_start:w_end] += grads[i, c_out, h, w] * self.W[c_out]

        if self.weight_decay:
            self.grads['W'] += 2 * self.weight_decay_lambda * self.W
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.softmax_output = None
        self.labels = None
        self.has_softmax = True
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels

        if self.has_softmax:
            max_p = np.max(predicts, axis=1, keepdims=True)
            shifted_logits = predicts - max_p  
            exp_p = np.exp(shifted_logits)
            self.softmax_output = exp_p / np.sum(exp_p, axis=1, keepdims=True)
        else:
            self.softmax_output = predicts

        one_hot = np.eye(self.max_classes)[labels]
        loss = -np.sum(one_hot * np.log(self.softmax_output + 1e-10)) / len(labels)

        reg_loss = 0
        for layer in self.model.layers:
            if isinstance(layer, L2Regularization):
                reg_loss += layer.l2_loss

        return loss + reg_loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        batch_size = len(self.labels)
        one_hot =  np.eye(self.max_classes)[self.labels]
        grad = (self.softmax_output - one_hot) / batch_size
        # Then send the grads to model for back propagation
        self.model.backward(grad)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class MSELoss(Layer):
    def __init__(self, model=None, max_classes=10):
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.softmax_output = None
        self.labels = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        Compute the mean squared error loss.
        
        Args:
            predicts: [batch_size, D]
            labels: [batch_size,]
        Returns:
            loss: scalar MSE loss value
        """

        self.labels = labels

        max_p = np.max(predicts, axis=1, keepdims=True)
        shifted_logits = predicts - max_p  
        exp_p = np.exp(shifted_logits)
        self.softmax_output = exp_p / np.sum(exp_p, axis=1, keepdims=True)
        
        one_hot = np.eye(self.max_classes)[labels]
        loss = np.mean(np.square(self.softmax_output - one_hot))
        
        reg_loss = 0
        if self.model is not None:
            for layer in self.model.layers:
                if isinstance(layer, L2Regularization):
                    reg_loss += layer.l2_loss
        
        return loss + reg_loss
    
    def backward(self):
        """
        Compute the gradient of MSE loss with respect to the inputs.
        The gradient is (2/N) * (predicts - labels)
        """
        batch_size = self.labels.shape[0]
        one_hot = np.eye(self.max_classes)[self.labels]

        grad = (2.0 / batch_size) * (self.softmax_output - one_hot)
        self.model.backward(grad)


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, layer, reg_coeff=0.01):
        super().__init__()
        self.layer = layer        
        self.reg_coeff = reg_coeff  
        self.l2_loss = 0.0          
        self.params = {'W': layer.W, 'b': layer.b}
        self.weight_decay = layer.weight_decay
        self.weight_decay_lambda = layer.weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        output = self.layer(X)
        self.l2_loss = 0.0
        
        for param in self.layer.params.values():
            self.l2_loss += np.sum(param ** 2) 
            
        self.l2_loss = 0.5 * self.reg_coeff * self.l2_loss  
            
        return output

    def backward(self, grad):
        reg_grads = {}
        for name, param in self.layer.params.items():
            reg_grads[name] = self.reg_coeff * param  

        for name in self.layer.grads.keys():
            if self.layer.grads[name] is not None:
                self.layer.grads[name] += reg_grads[name]
            else:
                self.layer.grads[name] = reg_grads[name]
                
        return self.layer.backward(grad)
    
    def clear_grad(self):
        self.layer.clear_grad()

       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.optimizable = False
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch, channels, H, W = X.shape
        k = self.kernel_size
        H_out = (H - k) // self.stride + 1
        W_out = (W - k) // self.stride + 1
        
        output = np.zeros((batch, channels, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k
                region = X[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(region, axis=(2,3))
        return output

    def backward(self, grad):
        X = self.input
        batch, channels, H, W = X.shape
        k = self.kernel_size
        H_out = grad.shape[2]
        W_out = grad.shape[3]
        
        dX = np.zeros_like(X)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k
                
                region = X[:, :, h_start:h_end, w_start:w_end]
                max_mask = (region == np.max(region, axis=(2,3), keepdims=True))
                dX[:, :, h_start:h_end, w_start:w_end] += max_mask * grad[:, :, i:i+1, j:j+1]
        return dX

class Flatten(Layer):
    """展平层"""
    def __init__(self):
        super().__init__() 
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
    
class EarlyStopping:
    def __init__(self, 
                 monitor,
                 patience=10,                  
                 min_delta=1e-4,
                 restore_best_weights=True):

        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.restore_best = restore_best_weights
        
        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None

    def check_improvement(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return True
        else:
            return current_value >= (self.best_value - self.min_delta)

    def step(self, current_value, model):
        if self.check_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = self.counter
            self.counter = 0  
            if self.restore_best:
                self.best_weights = [layer.params.copy() if hasattr(layer, 'params') else None for layer in model.layers]
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_model(self, model):
        if self.best_weights is not None:
            for layer, weights in zip(model.layers, self.best_weights):
                if hasattr(layer, 'params'):
                    layer.params = weights.copy()

def data_aug(img, type):
    data = img.copy()
    data = np.reshape(data, (-1, 28, 28))
    if type == 'shift':
        idx = np.random.permutation(np.arange(data.shape[0]))[:10000]
        for i in idx:
            tx, ty = np.random.randint(-3, 4, 2)
            if tx > 0:
                for col in range(0, 28, -1):
                    if col >= tx:
                        data[i][:][col] = data[i][:][col-tx]
                    else:
                        data[i][:][col] = 0
            elif tx < 0:
                tx = -tx
                for col in range(0, 28):
                    if col < 28 - tx:
                        data[i][:][col] = data[i][:][col+tx]
                    else:
                        data[i][:][col] = 0
            if ty > 0:
                for row in range(0, 28):
                    if row < 28 - ty:
                        data[i][row][:] = data[i][row+ty][:]
                    else:
                        data[i][row][:] = 0
            elif ty < 0:
                ty = -ty
                for row in range(0, 28, -1):
                    if row >= ty:
                        data[i][row][:] = data[i][row-ty][:]
                    else:
                        data[i][row][:] = 0

    data = np.reshape(data, (-1, 28*28))
    return data