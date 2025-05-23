import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic training data
num_samples = 100
X = np.random.uniform(0, 10, size=(num_samples, 1))  # Study hours from 0 to 10
true_slope = 5
true_intercept = 50
noise = np.random.normal(0, 5, size=(num_samples, 1))  # Add noise to simulate real data

y = true_slope * X + true_intercept + noise  # Exam score = slope * hours + intercept + noise

# Step 2: Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Step 3: Visualize the data and the learned regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Training Data", alpha=0.7)
#plt.plot(X, predictions, color='red', label='Learned Regression Line', linewidth=2)
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Print out the learned parameters
#print(f"Learned slope: {model.coef_[0][0]:.2f}")
#print(f"Learned intercept: {model.intercept_[0]:.2f}")
