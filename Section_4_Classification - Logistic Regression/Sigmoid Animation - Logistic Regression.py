# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load the Dataset
df = pd.read_csv("Logistic_Regression_E-Commerce_Behavior_Dataset.csv")

# 4. Prepare Features and Target
X = df[['Time_on_Site', 'Pages_Viewed', 'Referred_by_Ad', 'Previous_Purchases', 'Location_Score']]
y = df['Purchase_Made']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

coefficients.sort_values(by='Coefficient', ascending=False)

# 7. Probability Threshold Visualization

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Fit logistic regression using only "Pages_Viewed"
X_single = df[['Pages_Viewed']]
y = df['Purchase_Made']

model_single = LogisticRegression()
model_single.fit(X_single, y)

# Set up plot for animation
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 6))

# Generate prediction range
x_range = np.linspace(X_single.min().values[0], X_single.max().values[0], 300).reshape(-1, 1)
probs = model_single.predict_proba(x_range)[:, 1]

# Plot initial elements
scatter = ax.scatter(X_single, y, alpha=0.3, c=y, cmap='bwr', label='Actual Data')
line, = ax.plot(x_range, probs, color='blue', label='Sigmoid Output (P(Purchase))')
threshold_line = ax.axhline(0.5, color='red', linestyle='--')
threshold_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right',
                         fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# Formatting
ax.set_xlabel("Pages Viewed")
ax.set_ylabel("Probability of Purchase")
ax.set_title("Sigmoid and Moving Decision Threshold")
ax.set_ylim(-0.1, 1.1)
ax.legend()
ax.grid(True)

# Animation update function
def update(frame):
    threshold = frame / 100
    threshold_line.set_ydata([threshold, threshold])  # ← FIXED: y must be a list
    threshold_text.set_text(f'Threshold = {threshold:.2f}')
    return threshold_line, threshold_text

# Run animation
ani = animation.FuncAnimation(fig, update, frames=range(20, 90, 2), interval=200, blit=False)
plt.show()