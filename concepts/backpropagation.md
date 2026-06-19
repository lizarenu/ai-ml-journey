# Backpropagation

## Intuition

`y = mx + c` is the basic function for linear scenarios. To represent nonlinear scenarios, we use neural networks.

A neural network: take inputs, pass them through layers, get an output.

**Hidden layers** — each hidden layer is one pass through a linear function, then an activation function.

- Hidden layers are made of neurons/nodes. Each neuron's linear function holds a weight per input feature, initialized to different values per neuron — this is the network trying to learn a useful weight for each feature.
- The result of the linear step (`Z`) goes through an **activation function**. The activation transforms `Z` (the output of the linear combination), not the weights themselves — the weights stay as linear coefficients. This is what injects nonlinearity into the values flowing forward, and that nonlinear result (`A`) becomes the next layer's input.
- This nonlinearity is what stops multiple stacked linear functions from collapsing into one single linear equation — without it, depth would add no power at all.
- Layer by layer, this repeats. The *cumulative* effect across all layers is what eventually warps the input space enough that the final output layer can separate the classes with something as simple as a line.

**Why multiple (smaller) layers instead of one big layer:** with multiple layers, each layer builds new weights off of the previous layer's activation outputs — producing more expressive results. Without this, you'd need a very large number of neurons in a single layer to capture all the needed weight combinations, and even then it may not work, especially for nonlinear cases.

**Output layer:** gives the final result/prediction of the network. This result is then used to calculate the **loss function** value, by comparing the prediction to the actual target value — this is how we evaluate how well the network is doing.

## Derivatives, partial derivatives, and why we need them

The value of a derivative/partial derivative — taken with respect to a **parameter** (weight or bias), holding everything else constant — is that it tells us how that specific parameter is affecting the Loss. We use this to adjust the parameter in a direction that reduces the Loss.

The loss, when plotted against all the parameters, can have multiple local minima (beyond scope for now). To get to a minimum, we have to keep adjusting the parameters — that's **gradient descent**:

```
parameter = parameter - (dL/dparameter) * jump

```
the change is a subtraction because we want to move in the opposite direction of the gradient (which points to the direction of the steepest increase of the function/loss). so if the derivative is positive - then we want to reduce the parameter and vice versa. 

`jump` is the step size (learning rate).

To calculate all the partial derivatives needed for this update, we move backward through the network starting from the output — this is **backpropagation**.

The forward pass is the computational graph of inputs moving through the hidden layers to the output.

We run forward pass → backward pass → parameter update, repeated some number of times — these are **epochs**. The exact structure of an epoch (one update per epoch vs. multiple, via mini-batching) can vary depending on the situation/dataset size.

## Rules for computing the derivatives

- **Chain rule on a computational graph:** multiply local derivatives along a single path; add contributions when a variable's path splits into multiple downstream paths.
- Each node in the graph only has its own *local* derivative — the derivative of its output with respect to the inputs it directly receives. Chaining these local derivatives together (multiply along a path, add across paths) is what gives you the full gradient back to any earlier variable.
- **Matrix multiplication shape rule:** `(a,b) @ (b,c) = (a,c)`. Use transpose when needed to make shapes line up.
- **Gradient shape rule:** the derivative of a scalar loss with respect to any variable must have the *same shape* as that variable.

## What generalizes vs. what's situational

The core linear form `Wx + b` doesn't change — but constraints can be added to it (e.g. CNNs share weights + restrict to local connectivity; RNNs reuse the same weights recurrently across time steps).

What varies by problem and has to be re-derived each time:
- the activation function (sigmoid, ReLU, etc.)
- the loss function
- the sizes/shapes of everything (depends on layer widths, batch size)

## Code

`drills/backprop_from_scratch.py` — `forward()` and `backward()` for a 2-layer XOR network (sigmoid + MSE), trained and verified converging.

## Gotchas

- Sigmoid's derivative is `a(1-a)` in terms of its *output* `a`, not `z(1-z)` in terms of its input `z`.
- Transpose direction for `dW2`: should be `A1.T @ dZ2`, not `A1 @ dZ2`. The gradient-shape rule above catches this immediately if you check shapes.
- `*` (elementwise) vs `@` (matmul) — easy to use the wrong one.
