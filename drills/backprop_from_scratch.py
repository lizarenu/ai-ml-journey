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


def train_momentum(X, y, n_hidden=4, lr=0.5, beta=0.9, epochs=10000):
    """
    Same as train(), but updates W1, b1, W2, b2 using momentum
    instead of plain SGD.

    Maintain one velocity accumulator per parameter (vW1, vb1, vW2, vb2),
    each the same shape as its corresponding parameter, initialized to zero.

    Each epoch:
      1. forward pass
      2. backward pass to get dW1, db1, dW2, db2
      3. update each velocity:  v = beta * v + grad
      4. update each parameter: param = param - lr * v

    returns: trained W1, b1, W2, b2
    """
    W1 = np.random.randn(X.shape[1],n_hidden ) * 0.1
    b1 = np.zeros((1,n_hidden ))
    W2 = np.random.randn(n_hidden,1 ) * 0.1
    b2 = np.zeros((1,1 ))
    vW1 = np.zeros((X.shape[1],n_hidden ))
    vW2 = np.zeros((n_hidden,1 ))
    vb1 = np.zeros((1,n_hidden ))
    vb2 = np.zeros((1,1 ))


    for i in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2, W2)
        vW1 = beta * vW1 + (1 - beta) * dW1
        vW2 = beta * vW2 + (1 - beta) * dW2
        vb1 = beta * vb1 + (1 - beta) * db1
        vb2 = beta * vb2 + (1 - beta) * db2
        W1 = W1 - lr* vW1
        b1 = b1 - lr* vb1
        W2 = W2 - lr* vW2
        b2 = b2 - lr* vb2


    return W1, b1, W2, b2


def train_rmsprop(X, y, n_hidden=4, lr=0.5, beta2=0.99, eps=1e-8, epochs=10000):
    """
    Same as train(), but updates W1, b1, W2, b2 using RMSprop
    instead of plain SGD.

    Maintain one squared-gradient accumulator per parameter
    (sW1, sb1, sW2, sb2), each the same shape as its corresponding
    parameter, initialized to zero.

    Each epoch:
      1. forward pass
      2. backward pass to get dW1, db1, dW2, db2
      3. update each accumulator: s = beta2 * s + (1 - beta2) * grad**2
      4. update each parameter:   param = param - lr * grad / (sqrt(s) + eps)

    returns: trained W1, b1, W2, b2
    """
    W1 = np.random.randn(X.shape[1],n_hidden ) * 0.1
    b1 = np.zeros((1,n_hidden ))
    W2 = np.random.randn(n_hidden,1 ) * 0.1
    b2 = np.zeros((1,1 ))
    sW1 = np.zeros((X.shape[1],n_hidden ))
    sW2 = np.zeros((n_hidden,1 ))
    sb1 = np.zeros((1,n_hidden ))
    sb2 = np.zeros((1,1 ))


    for i in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2, W2)
        sW1 = beta2 * sW1 + (1-beta2)*dW1**2
        sW2 = beta2 * sW2 + (1-beta2)*dW2**2
        sb1 = beta2 * sb1 + (1-beta2)*db1**2
        sb2 = beta2 * sb2 + (1-beta2)*db2**2
        W1 = W1 - lr* dW1 / (np.sqrt(sW1) + eps)
        b1 = b1 - lr* db1 / (np.sqrt(sb1) + eps)
        W2 = W2 - lr* dW2 / (np.sqrt(sW2) + eps)
        b2 = b2 - lr* db2 / (np.sqrt(sb2) + eps)


    return W1, b1, W2, b2


def train_adam(X, y, n_hidden=4, lr=0.5, beta1=0.9, beta2=0.999, eps=1e-8, epochs=10000):
    """
    Same as train(), but updates W1, b1, W2, b2 using Adam
    instead of plain SGD.

    Maintain, per parameter, a first-moment accumulator (mW1, mb1, mW2, mb2)
    and a second-moment accumulator (vW1, vb1, vW2, vb2), all initialized
    to zero, same shape as their corresponding parameter.

    Each epoch (t starts at 1, not 0 -- needed for bias correction):
      1. forward pass
      2. backward pass to get dW1, db1, dW2, db2
      3. update each first moment:  m = beta1 * m + (1 - beta1) * grad
      4. update each second moment: v = beta2 * v + (1 - beta2) * grad**2
      5. bias-correct:  m_hat = m / (1 - beta1**t), v_hat = v / (1 - beta2**t)
      6. update each parameter:     param = param - lr * m_hat / (sqrt(v_hat) + eps)

    returns: trained W1, b1, W2, b2
    """
    W1 = np.random.randn(X.shape[1],n_hidden ) * 0.1
    b1 = np.zeros((1,n_hidden ))
    W2 = np.random.randn(n_hidden,1 ) * 0.1
    b2 = np.zeros((1,1 ))
    mW1 = np.zeros((X.shape[1],n_hidden ))
    mW2 = np.zeros((n_hidden,1 ))
    mb1 = np.zeros((1,n_hidden ))
    mb2 = np.zeros((1,1 ))
    vW1 = np.zeros((X.shape[1],n_hidden ))
    vW2 = np.zeros((n_hidden,1 ))
    vb1 = np.zeros((1,n_hidden ))
    vb2 = np.zeros((1,1 ))


    for i in range(1,epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2, W2)

        vW1 = beta2 * vW1 + (1-beta2)*dW1**2
        vW2 = beta2 * vW2 + (1-beta2)*dW2**2
        vb1 = beta2 * vb1 + (1-beta2)*db1**2
        vb2 = beta2 * vb2 + (1-beta2)*db2**2

        vW1_hat = vW1 / (1-beta2**i)
        vW2_hat = vW2 / (1-beta2**i)
        vb1_hat = vb1 / (1-beta2**i)
        vb2_hat = vb2 / (1-beta2**i)

        mW1 = beta1 * mW1 + (1-beta1)*dW1
        mW2 = beta1 * mW2 + (1-beta1)*dW2
        mb1 = beta1 * mb1 + (1-beta1)*db1
        mb2 = beta1 * mb2 + (1-beta1)*db2

        mW1_hat = mW1 / (1-beta1**i)
        mW2_hat = mW2 / (1-beta1**i)
        mb1_hat = mb1 / (1-beta1**i)
        mb2_hat = mb2 / (1-beta1**i)


        W1 = W1 - lr* mW1_hat / (np.sqrt(vW1_hat) + eps)
        b1 = b1 - lr* mb1_hat / (np.sqrt(vb1_hat) + eps)
        W2 = W2 - lr* mW2_hat / (np.sqrt(vW2_hat) + eps)
        b2 = b2 - lr* mb2_hat / (np.sqrt(vb2_hat) + eps)


    return W1, b1, W2, b2


""" Test Script """

optimizers = {
    "sgd": dict(fn=train, lr=0.5),
    "momentum": dict(fn=train_momentum, lr=0.5),
    "rmsprop": dict(fn=train_rmsprop, lr=0.5),
    "adam": dict(fn=train_adam, lr=0.1),  # adam needs a smaller lr than the others
}

for name, cfg in optimizers.items():
    np.random.seed(0)
    W1, b1, W2, b2 = cfg["fn"](X, y, lr=cfg["lr"], epochs=5000)
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    loss = np.mean((A2 - y) ** 2)
    preds = (A2 > 0.5).astype(int).flatten().tolist()
    print(f"{name:9s} loss={loss:.5f}  preds={preds}  target={y.flatten().tolist()}")

