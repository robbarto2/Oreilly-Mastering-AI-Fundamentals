import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Step 1: Generate Synthetic Data
np.random.seed(42)
n_samples = 300
pages_viewed = np.random.poisson(lam=4, size=n_samples)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# True purchase probability increases with pages viewed
true_probs = sigmoid(1.2 * (pages_viewed - 5))
purchases = np.random.binomial(1, true_probs)

X = pages_viewed.reshape(-1, 1)
y = purchases
X_norm = (X - X.mean()) / X.std()

# Step 2: Train Logistic Regression with Gradient Descent
epochs = 100
lr = 0.1
w, b = 0.0, 0.0
w_list, b_list = [], []

for _ in range(epochs):
    z = w * X_norm.flatten() + b
    pred = sigmoid(z)
    error = pred - y
    grad_w = np.dot(error, X_norm.flatten()) / n_samples
    grad_b = np.sum(error) / n_samples
    w -= lr * grad_w
    b -= lr * grad_b
    w_list.append(w)
    b_list.append(b)

# Step 3: Set up the Animation
x_plot = np.linspace(X_norm.min(), X_norm.max(), 300)
fig, ax = plt.subplots(figsize=(8, 5))

def animate(i):
    ax.clear()
    ax.scatter(X_norm, y, alpha=0.3, c=y, cmap='bwr', label='Actual Data')
    y_curve = sigmoid(w_list[i] * x_plot + b_list[i])
    ax.plot(x_plot, y_curve, color='blue', label=f'Epoch {i+1}')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1)
    ax.set_title("Training Logistic Regression with Gradient Descent")
    ax.set_xlabel("Pages Viewed (normalized)")
    ax.set_ylabel("Predicted Probability")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True)

ani = FuncAnimation(fig, animate, frames=epochs, interval=60)

# Step 4: Save Animation to GIF
ani.save("logistic_regression_training.mp4", writer="ffmpeg", fps=15)