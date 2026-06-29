# Gradient Descent Variants and Regularization

## Why vanilla GD struggles

Features can be on varying scales — e.g. (1–10) vs (10,000–999,999). The sensitivity of the loss function to each weight varies because of these feature scales. A change in the weight of a large-scale feature creates a large change in the computed loss — it has steeper curvature when the loss is plotted against that weight. A small-scale feature has flatter curvature.

When you plot the loss contours in weight-space, this produces elongated ellipses rather than circular bowls — steep in one direction, flat in another.

The gradient points in the direction of steepest increase. On an elliptical surface, this means it points across the narrow (steep) axis and barely moves along the flat axis — which is not in alignment with the direction toward the local minima. Repeated steps create a zigzag path: overshooting back and forth across the steep axis while crawling slowly along the flat axis toward the minimum.

## Momentum

Momentum keeps a running average (velocity) of past gradients instead of using the raw current gradient.

```
m_t = beta * m_(t-1) + (1 - beta) * dw
w = w - lr * m_t
```

The `(1 - beta)` ensures the velocity is a weighted average rather than an unbounded accumulation — beta controls how much history carries over (typically ~0.9).

Two effects:
- **Flat direction:** gradients consistently point the same way, so they reinforce each other in the average — velocity builds up and effective steps get larger. (Acceleration)
- **Steep direction:** gradients keep flipping sign, so they cancel each other in the average — velocity stays small and oscillation is damped. (Dampening)

## RMSprop

Instead of one global learning rate, RMSprop gives each weight its own effective learning rate based on its gradient history. If a weight has historically had large gradients (steep direction), shrink its step. If small gradients (flat direction), allow a larger effective step.

This is tracked using a running average of the squared gradient:

```
s_t = beta * s_(t-1) + (1 - beta) * dw^2
w = w - lr * dw / (sqrt(s_t) + epsilon)
```

Squaring removes the sign so past gradient direction doesn't cancel out — we're measuring magnitude. `epsilon` (e.g. 1e-8) prevents division by zero when `s_t` is very small. The `dw` in the numerator is still needed — `lr / sqrt(s_t)` is just a scaled learning rate; `dw` is what provides the actual direction and magnitude of the step.

## Adam

Adam combines both approaches: momentum (tracks mean of gradients) and RMSprop (tracks mean of squared gradients), with a bias correction.

```
m_t = beta1 * m_(t-1) + (1 - beta1) * dw       (1st moment — momentum)
v_t = beta2 * v_(t-1) + (1 - beta2) * dw^2      (2nd moment — RMSprop)

m_hat = m_t / (1 - beta1^t)                     (bias correction)
v_hat = v_t / (1 - beta2^t)

w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Both `m_t` and `v_t` are initialized at 0. In early iterations there is little gradient history, so the estimates are biased toward 0. Dividing by `(1 - beta^t)` corrects for this — at early steps `beta^t` is close to 1 so the correction inflates the estimate significantly; as `t` grows `beta^t` approaches 0 and the correction fades out. (**Bias Correction**)

## Regularization

If the weights for any feature are inflated, the model is most likely overfitting — fitting the noise of the training set rather than the underlying pattern. This is the bias-variance tradeoff applied to neural network weights.

To handle this, a penalty linked to weight sizes is added to the loss:

```
Loss_total = Loss_data + lambda * penalty(weights)
```

`lambda` controls strictness — high lambda means high bias, low variance; low lambda means less bias, higher variance risk.

Two common penalty choices:

**L1 (Lasso):** penalty = sum of |w|
**L2 (Ridge):** penalty = sum of w^2

Taking the gradient of each penalty with respect to a weight shows the difference:
- L1 gradient = +1 or -1 (constant, regardless of weight size)
- L2 gradient = 2w (proportional to the weight)

For L2: as the weight decreases, the penalty gradient also decreases — so the pull toward zero weakens as the weight shrinks. It never fully reaches zero.

For L1: the pull is constant regardless of how small the weight gets. When the weight becomes smaller than the step size, the update overshoots zero. The sign flips, the update pulls back — the weight pins around zero. In practice this drives weights to effectively zero, producing sparse weight vectors (feature selection).
