import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # shape (4, 2)
y = np.array([[0], [1], [1], [0]])              # shape (4, 1)


def sigmoid(z):
    """
    z: ndarray, any shape
    returns: ndarray, same shape as z. Elementwise sigmoid(z) = 1/(1+e^-z)
    """
    # np.exp applies e^x elementwise to every entry of the array at once —
    # no loop needed, this is what "vectorized" means in numpy.
    return 1 / (1 + np.exp(-z))


def forward(X, W1, b1, W2, b2):
    """
    Run the forward pass through the 2-layer network.

    X:  (n_samples, n_features)
    W1: (n_features, n_hidden)
    b1: (1, n_hidden)
    W2: (n_hidden, 1)
    b2: (1, 1)

    returns: Z1, A1, Z2, A2
        Z1: (n_samples, n_hidden)  -- pre-activation of hidden layer
        A1: (n_samples, n_hidden)  -- sigmoid(Z1)
        Z2: (n_samples, 1)         -- pre-activation of output layer
        A2: (n_samples, 1)         -- sigmoid(Z2), the prediction
    """
    Z1 = X @ W1 + b1

    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2

    A2 = sigmoid(Z2)

    return Z1,A1,Z2,A2




def backward(X, y, Z1, A1, Z2, A2, W2):
    """
    Compute gradients of the mean-squared-error loss w.r.t. every parameter,
    using the chain rule you just derived by hand (extended to matrices).

    Loss: L = mean((A2 - y)^2)

    returns: dW1, db1, dW2, db2 -- same shapes as W1, b1, W2, b2

    Hint: this is the same local-derivative chain as the single neuron
    (dL/da2, da2/dz2, dz2/dW2, ...) just applied with matrix shapes.
    db1 and db2 should sum gradients over the sample axis (axis=0),
    keeping dims, since b is broadcast across samples in the forward pass.
    """

    n = X.shape[0]
    dZ2 = (2/n) * (A2-y) * A2 * (1-A2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis = 0, keepdims=True)
    dA1 = dZ2 @ W2.T 
    dZ1 = dA1 * A1 * (1-A1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis = 0, keepdims=True)


    return dW1, db1, dW2, db2




def train(X, y, n_hidden=4, lr=0.5, epochs=10000):
    """
    Initialize W1, b1, W2, b2 (small random values for weights, zeros for biases),
    then repeatedly: forward pass -> backward pass -> gradient descent update,
    for `epochs` iterations.

    returns: trained W1, b1, W2, b2
    """
    W1 = np.random.randn(X.shape[1],n_hidden ) * 0.1
    b1 = np.zeros((1,n_hidden ))
    W2 = np.random.randn(n_hidden,1 ) * 0.1
    b2 = np.zeros((1,1 ))


    for i in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2, W2)
        W1 = W1 - lr* dW1
        b1 = b1 - lr* db1
        W2 = W2 - lr* dW2
        b2 = b2 - lr* db2


    return W1, b1, W2, b2


""" Test Script """ 

W1, b1, W2, b2 = train(X,y)

Z1,A1,Z2,A2 = forward(X, W1, b1, W2, b2)

for i in range(y.shape[0]):
    print(y[i], 1 if A2[i] > 0.5 else 0)

