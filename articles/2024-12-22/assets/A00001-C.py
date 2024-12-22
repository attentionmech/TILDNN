import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim

# Define XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)

# Convert to tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).view(-1, 1)

# Define a simple neural network for XOR
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize model, loss function, and optimizer
model = XORModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Prepare for animation
epochs = 500
losses = []
weights_history = []
activation_history = []
error_surface_history = []

# Training loop with data visualization
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
ax1, ax2, ax3 = axs[0]
ax4, ax5, ax6 = axs[1]

# Prepare to record the animation frames
def update(frame):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    # Record loss
    losses.append(loss.item())

    # Record weights for visualization
    weights_history.append([model.fc1.weight.detach().numpy(), model.fc2.weight.detach().numpy()])
    
    # Record activations
    with torch.no_grad():
        activations = model.sigmoid(model.fc1(X_tensor)).detach().numpy()
        activation_history.append(activations)

    # Plot XOR data and decision boundary
    ax1.clear()
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', s=100, edgecolors='black')
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid_preds = model(grid_points).detach().numpy().reshape(xx.shape)
    ax1.contourf(xx, yy, grid_preds, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.5)
    ax1.set_title("XOR Dataset & Decision Boundary")

    # Plot loss curve
    ax2.clear()
    ax2.plot(losses, label='Loss')
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    # Plot weights history
    ax3.clear()
    ax3.plot([w[0][0, 0] for w in weights_history], label="Weight 1", color='red')
    ax3.plot([w[0][0, 1] for w in weights_history], label="Weight 2", color='blue')
    ax3.plot([w[1][0, 0] for w in weights_history], label="Weight 3", color='green')
    ax3.plot([w[1][0, 1] for w in weights_history], label="Weight 4", color='purple')
    ax3.set_title("Weight History")
    ax3.legend()

    # Plot activations
    ax4.clear()
    ax4.scatter(activation_history[-1][:, 0], activation_history[-1][:, 1], c=y, cmap='coolwarm', edgecolors='black')
    ax4.set_title("Activations of Hidden Layer Neurons")

    # Plot error heatmap
    ax5.clear()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid_preds = model(grid_points).detach().numpy().reshape(xx.shape)
    ax5.contourf(xx, yy, np.abs(grid_preds - 0.5), cmap='hot', levels=20)
    ax5.set_title("Error Heatmap")

    # Show epoch number
    ax6.clear()
    ax6.text(0.5, 0.5, f'Epoch: {frame}/{epochs}', ha='center', va='center', fontsize=12)
    ax6.axis('off')

    return ax1, ax2, ax3, ax4, ax5, ax6

# Create the animation
ani = FuncAnimation(fig, update, frames=epochs, blit=False)

# Save the animation as an MP4
ani.save('xor_learning.mp4', writer='ffmpeg', fps=30)

# Display the plot
plt.tight_layout()
plt.show()

