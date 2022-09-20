import numpy as np


def random_unbias_weight(dim1,dim2):
        return np.random.randn(dim1,dim2)* 1./(np.sqrt(dim1))

def random_unbias_bias(dim):
    return np.zeros((1,dim))

class NeuralNetwork:
    def __init__(self,input_layer_dim, hidden_layer_dim, output_layer_dim,
                    activation,output_activation):
        self.input_layer_dim = input_layer_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.output_layer_dim = output_layer_dim
        self.activation = activation
        self.output_activation = output_activation
        self.params = {}
        self.back_prop = {}
        self.gradients = {}
        self.initialize_weight()

    def initialize_weight(self):
        self.params = {
            'W1': random_unbias_weight(self.input_layer_dim,self.hidden_layer_dim),
            'b1': random_unbias_bias(self.hidden_layer_dim),
            'W2': random_unbias_weight(self.hidden_layer_dim,self.output_layer_dim),
            'b2': random_unbias_bias(self.output_layer_dim)
        }
    
    def forward_pass(self,X):
        # z1 -> a1 -> z2 -> output
        z1 = np.matmul(X,self.params['W1']) + self.params['b1']
        # print(self.params['W1'][self.params['W1']>0])
        a1 = self.activation.forward(z1)
        z2 = np.matmul(a1,self.params['W2']) + self.params['b2']
        output = self.output_activation.forward(z2)
        self.back_prop = {
            'X':X,
            'z1':z1,
            'a1':a1,
            'z2':z2,
            'output':output,
        }
        return output

    def backward_pass(self,Y):
        batch = Y.shape[0]
        norm = 1. / batch
      
        dZ2 = self.output_activation.backward(self.back_prop['output'], Y)
        dW2 = norm * np.matmul(self.back_prop['a1'].T,dZ2)
        db2 = norm * np.sum(dZ2, axis=0, keepdims=True)
        self.gradients['W2'] = dW2
        self.gradients['b2'] = db2

        dA1 = np.matmul(dZ2, self.params['W2'].T)
        dZ1 = dA1*self.activation.backward(self.back_prop['z1'])
        dW1 = norm * np.matmul(self.back_prop['X'].T, dZ1)
        db1 = norm * np.sum(dZ1, axis=0, keepdims=True)
        self.gradients['W1'] = dW1
        self.gradients['b1'] = db1




