class Layer:
    def __init__(self):
        pass
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output, learning_rate,lambda_=0.0):
        raise NotImplementedError