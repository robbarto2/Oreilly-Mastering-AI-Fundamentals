import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate synthetic data
np.random.seed(42)
num_samples = 100
true_slope = 2.5
true_intercept = 8
x_data = np.random.rand(num_samples, 1) * 3
y_data = true_slope * x_data + true_intercept + np.random.randn(num_samples, 1)

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# Define a linear model
model = nn.Linear(1, 1)

# Initialize weights and bias to 0
with torch.no_grad():
    model.weight.fill_(0.0)
    model.bias.fill_(0.0)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Store weights and biases for each epoch
w_history = []
b_history = []

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        w = model.weight.item()
        b = model.bias.item()
        w_history.append(w)
        b_history.append(b)

# Set up animation
fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='Data')
line, = ax.plot([], [], 'r-', label='Prediction')
ax.set_xlim(x_data.min(), x_data.max())
ax.set_ylim(y_data.min(), y_data.max())
ax.set_title("Gradient Descent Convergence")
ax.legend()

# Set up animation
fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='Data')
line, = ax.plot([], [], 'r-', label='Prediction')
ax.set_xlim(x_data.min(), x_data.max())
ax.set_ylim(y_data.min(), y_data.max())
ax.set_title("Gradient Descent Convergence")
ax.legend()

# Add epoch counter text
epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

def animate(i):
    w = w_history[i]
    b = b_history[i]
    x_vals = np.linspace(x_data.min(), x_data.max(), 100)
    y_vals = w * x_vals + b
    line.set_data(x_vals, y_vals)
    epoch_text.set_text(f'Epoch: {i+1}/{epochs}')
    return line, epoch_text

ani = FuncAnimation(fig, animate, frames=len(w_history), interval=100, blit=True, repeat=False)

plt.show()

