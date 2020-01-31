import numpy as np

class SVM(object):
    """
    Support vector machine in numpy
    """
    def __init__(self, learning_rate, epochs):
        self.l_rate = learning_rate
        self.epochs = epochs
        
    def fit(self, x, y):
        weights = np.zeros(len(x[0]))
        output = []
        for epochs in range(self.epochs):
            for i, val in enumerate(x):
                if (y[i]*np.dot(x[i],weights) < 1):
                    weights = weights + self.l_rate * ((y[i]*x[i]) - (2*(1/self.epochs)*weights))
                else:
                    weights = weights + self.l_rate * (-2*(1/self.epochs)*weights)
        
        for i, val in enumerate(x):
            output.append(np.dot(x[i],weights))
            
        return weights, output
    
