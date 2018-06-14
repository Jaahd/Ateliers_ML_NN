
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

def sigmoid(x):
    """ sigmoid activation function """
    return 1 / (1 + np.exp(-x))

def tanh_prime(x):
    """ derivative of the tanh function """
    return 1 - np.tanh(x)

def train(X, Y, W1, W2, b1, b2):
    """ training function with both forward propagation and backward propagation """

    loss_history = []
    
    for epoch in range(epochs):

        # Forward propagation
        Z1 = np.add(np.dot(W1, X), b1)
        A1 = np.tanh(Z1)
        Z2 = np.add(np.dot(W2, A1), b2)  
        A2 = sigmoid(Z2)
        
        # Backward propagation
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(W2.T, dZ2), tanh_prime(Z1))
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
                
        # Parameter update
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        
        # Compute loss
        loss = np.mean(-np.add(np.multiply(Y,np.log(A2)),np.multiply(np.subtract(1, Y),np.log(np.subtract(1, A2)))))
        loss_history.append(loss)
    
    return loss_history, W1, W2, b1, b2

# loss_history, W1, W2, b1, b2 = train(X, Y, W1, W2, b1, b2)

def display(loss_history):

    plt.plot(loss_history)
    plt.show()
    print(loss_history[-1])

def predict(X, W1, W2, b1, b2):
    
    Z1 = np.add(np.dot(W1, X), b1)
    A1 = np.tanh(Z1)
    Z2 = np.add(np.dot(W2, A1), b2)  
    A2 = sigmoid(Z2)
      
    return np.array([0 if elt < 0.5 else 1 for elt in A2])
   
def test(W1, W2, b1, b2):    
    X = np.random.binomial(1, 0.5, (n_in,1))
    Y = X ^ 1
    Yhat = predict(X, W1, W2, b1, b2)
    return Y.T[0], Yhat.T

def check_perf(W1, W2, b1, b2):
    
    list_test = []
    for i in range(25):
        Y, Yhat = test(W1, W2, b1, b2)
        list_test.append(roc_auc_score(Y, Yhat))
        
    print(list_test)

if __name__ == "__main__":
    # var init
    # Num of neurones on each layers
    n_in = 10
    n_hidden = 100
    n_out = 10

    # Nb de 'training examples'
    m = 300

    alpha = 0.8  # Learning rate
    epochs = 500  # nb iterations du gradient descent

    # initialization of weigths and biaises
    W1 = np.random.randn(n_hidden, n_in) * 0.01 
    W2 = np.random.randn(n_out, n_hidden) * 0.01 
    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_out, 1))

    # Data generation
    X = np.random.binomial(1, 0.5, (n_in, m))
    Y = X ^ 1

    loss_history, W1, W2, b1, b2 = train(X, Y, W1, W2, b1, b2)
    check_perf(W1, W2, b1, b2)
