"""
Visual aid for the gradient descent zigzag / ill-conditioning discussion.
Loss surface: L(w1, w2) = a*w1^2 + b*w2^2, with b >> a (anisotropic curvature).
Generates two PNGs:
  1. gd_zigzag.png        - vanilla GD path on the elongated-contour surface
  2. gd_momentum_compare.png - same surface, vanilla GD vs momentum side by side
"""

import numpy as np
import matplotlib.pyplot as plt

a, b = 1.0, 25.0  # curvature along w1 (flat) vs w2 (steep)

def loss(w1, w2):
    return a * w1**2 + b * w2**2

def grad(w):
    return np.array([2 * a * w[0], 2 * b * w[1]])

def run_gd(start, lr, steps):
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad(w)
        path.append(w.copy())
    return np.array(path)

def run_momentum(start, lr, steps, beta=0.9):
    w = np.array(start, dtype=float)
    v = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        v = beta * v + grad(w)
        w = w - lr * v
        path.append(w.copy())
    return np.array(path)

start = [8.0, 1.0]
lr = 0.035
steps = 25

path_gd = run_gd(start, lr, steps)
path_mom = run_momentum(start, lr * 0.6, steps, beta=0.6)

# sign flips in w2 -> count of zigzag oscillations
flips = np.sum(np.diff(np.sign(path_gd[:, 1])) != 0)
print(f"Vanilla GD: {flips} sign flips in w2 over {steps} steps (zigzag count)")
print(f"Vanilla GD final point: {path_gd[-1]}, loss={loss(*path_gd[-1]):.4f}")
print(f"Momentum final point: {path_mom[-1]}, loss={loss(*path_mom[-1]):.4f}")

w1_range = np.linspace(-9, 9, 400)
w2_range = np.linspace(-2.5, 2.5, 400)
W1, W2 = np.meshgrid(w1_range, w2_range)
L = loss(W1, W2)

# --- Plot 1: vanilla GD zigzag alone ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.contour(W1, W2, L, levels=20, cmap='viridis', alpha=0.6)
ax.plot(path_gd[:, 0], path_gd[:, 1], 'o-', color='red', markersize=3, linewidth=1, label='vanilla GD path')
ax.plot(0, 0, 'k*', markersize=15, label='minimum')
ax.plot(start[0], start[1], 'gs', markersize=8, label='start')
ax.set_xlabel('w1 (flat direction)')
ax.set_ylabel('w2 (steep direction)')
ax.set_title('Vanilla GD zigzag on an ill-conditioned (elongated) loss surface')
ax.legend()
ax.set_aspect('auto')
fig.tight_layout()
fig.savefig('drills/visuals/gd_zigzag.png', dpi=130)
plt.close(fig)

# --- Plot 2: vanilla GD vs momentum side by side ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, path, title in zip(axes, [path_gd, path_mom], ['Vanilla GD', 'GD + Momentum']):
    ax.contour(W1, W2, L, levels=20, cmap='viridis', alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], 'o-', color='red', markersize=3, linewidth=1)
    ax.plot(0, 0, 'k*', markersize=15)
    ax.plot(start[0], start[1], 'gs', markersize=8)
    ax.set_xlabel('w1 (flat direction)')
    ax.set_title(title)
axes[0].set_ylabel('w2 (steep direction)')
fig.suptitle('Zigzag vs damped oscillation: momentum smooths the steep-axis overshoot')
fig.tight_layout()
fig.savefig('drills/visuals/gd_momentum_compare.png', dpi=130)
plt.close(fig)

print("Saved drills/visuals/gd_zigzag.png and drills/visuals/gd_momentum_compare.png")
