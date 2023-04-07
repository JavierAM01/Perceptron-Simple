import numpy as np

np.random.seed(0)


def sigmoid(x):
    return 1.0 / ( 1.0 + np.e**(-x) )

def mse(x1, x2):
    if type(x1) != list:
        x1, x2 = [x1], [x2]
    N = len(x1)
    return sum( (np.array(x1) - np.array(x2))**2 ) / N

class Perceptron:

    def __init__(self, n_inputs, lr):
        self.n = n_inputs
        self.lr = lr
        self.w = np.random.randn(n_inputs) # nº aleatorios distribuidos por una : N(0,1)
        self.w0 = np.random.randn(1)

    def forward(self, X):
        pred = np.dot(X, self.w) + self.w0
        return sigmoid(pred)
    
    def f(self, x, y):
        pred = self.forward(x)
        error = y - pred
        self.w  += self.lr * error * x
        self.w0 += self.lr * error
        return abs(error)

    def fit(self, X, Y, epochs=1):
        N = len(X)
        losses = []
        for epoch in range(epochs):
            for i in range(0,N):
                x = X[i,:] 
                y = Y[i]                   
                pred = self.forward(x)
                error = y - pred
                self.w  += self.lr * error * x
                self.w0 += self.lr * error
                losses.append(abs(error))
            print(f"[{epoch+1}] last error: {losses[-1]} | mean error: {sum(losses)/len(losses)}")
        return losses