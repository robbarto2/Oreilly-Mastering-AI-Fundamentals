import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional
from dataclasses import dataclass
import seaborn as sns

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 100
    learning_rate: float = 0.1
    early_stopping_patience: int = 10
    regularization_strength: float = 0.01
    convergence_threshold: float = 1e-4
    train_val_split: float = 0.8

def generate_synthetic_data(n_samples: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for logistic regression.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is normalized features and y is binary labels
    """
    np.random.seed(seed)
    pages_viewed = np.random.poisson(lam=4, size=n_samples)
    true_probs = sigmoid(1.2 * (pages_viewed - 5))
    purchases = np.random.binomial(1, true_probs)
    
    X = pages_viewed.reshape(-1, 1)
    X_norm = (X - X.mean()) / X.std()
    return X_norm, purchases

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow

def compute_gradients(X: np.ndarray, y: np.ndarray, w: float, b: float, 
                     reg_strength: float) -> Tuple[float, float]:
    """Compute gradients for logistic regression with regularization."""
    n_samples = len(y)
    z = w * X.flatten() + b
    pred = sigmoid(z)
    error = pred - y
    
    # Add L2 regularization
    grad_w = (np.dot(error, X.flatten()) / n_samples) + (reg_strength * w)
    grad_b = np.sum(error) / n_samples
    return grad_w, grad_b

def train_logistic_regression(X: np.ndarray, y: np.ndarray, config: TrainingConfig) -> Tuple[List[float], List[float], List[float]]:
    """Train logistic regression model using gradient descent.
    
    Returns:
        Lists of weights, biases, and losses during training
    """
    # Split data into train and validation sets
    train_size = int(len(X) * config.train_val_split)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    w, b = 0.0, 0.0
    w_list, b_list, loss_list = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training step
        grad_w, grad_b = compute_gradients(X_train, y_train, w, b, config.regularization_strength)
        w -= config.learning_rate * grad_w
        b -= config.learning_rate * grad_b
        
        # Compute validation loss
        val_pred = sigmoid(w * X_val.flatten() + b)
        val_loss = -np.mean(y_val * np.log(val_pred + 1e-15) + 
                          (1 - y_val) * np.log(1 - val_pred + 1e-15))
        
        # Early stopping
        if val_loss < best_val_loss - config.convergence_threshold:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
            
        w_list.append(w)
        b_list.append(b)
        loss_list.append(val_loss)
        
    return w_list, b_list, loss_list

def create_animation(X: np.ndarray, y: np.ndarray, w_list: List[float], 
                    b_list: List[float], loss_list: List[float], 
                    save_path: str = "logistic_regression_training.mp4") -> None:
    """Create and save animation of training process."""
    plt.style.use('bmh')  # Using a built-in style instead of seaborn
    x_plot = np.linspace(X.min(), X.max(), 300)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    fig.suptitle("Training Logistic Regression with Gradient Descent", fontsize=14)
    
    def animate(i):
        # Clear previous frame
        ax1.clear()
        ax2.clear()
        
        # Plot data and decision boundary
        ax1.scatter(X, y, alpha=0.3, c=y, cmap='coolwarm', label='Actual Data')
        y_curve = sigmoid(w_list[i] * x_plot + b_list[i])
        ax1.plot(x_plot, y_curve, color='blue', label=f'Decision Boundary')
        ax1.fill_between(x_plot, y_curve - 0.1, y_curve + 0.1, 
                        color='blue', alpha=0.1, label='Confidence Interval')
        ax1.axhline(0.5, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel("Pages Viewed (normalized)")
        ax1.set_ylabel("Predicted Probability")
        ax1.set_ylim(-0.1, 1.1)
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        # Plot loss curve
        ax2.plot(loss_list[:i+1], color='green')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss")
        ax2.grid(True)
        ax2.set_title(f"Current Loss: {loss_list[i]:.4f}")
        
        plt.tight_layout()
    
    ani = FuncAnimation(fig, animate, frames=len(w_list), interval=60)
    ani.save(save_path, writer="ffmpeg", fps=15)
    plt.close()

def main():
    """Main function to run the logistic regression training and visualization."""
    # Configuration
    config = TrainingConfig()
    
    # Generate and prepare data
    X, y = generate_synthetic_data()
    
    # Train model
    print("Training logistic regression model...")
    w_list, b_list, loss_list = train_logistic_regression(X, y, config)
    
    # Create animation
    print("Creating animation...")
    create_animation(X, y, w_list, b_list, loss_list)
    print("Animation saved as 'logistic_regression_training.mp4'")

if __name__ == "__main__":
    main() 