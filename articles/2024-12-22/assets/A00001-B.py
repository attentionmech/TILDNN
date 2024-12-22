import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import os

# XOR data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).view(-1, 1)

# Neural network model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 2 input features, 4 neurons in hidden layer
        self.fc2 = nn.Linear(4, 1)  # 4 hidden layer neurons, 1 output neuron
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Sigmoid activation for hidden layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output
        return x

# Initialize model, loss, and optimizer
model = XORModel()
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Create a directory to store images for the GIF
if not os.path.exists('images'):
    os.makedirs('images')

# Function to plot decision boundary and save as image
def plot_decision_boundary(X, y, model, epoch):
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    h = 0.01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=50, edgecolor='k', cmap=ListedColormap(['red', 'blue']))
    plt.title(f"Epoch {epoch}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')

    # Save the plot as an image
    plt.savefig(f'images/epoch_{epoch}.png')
    plt.close()

# Training loop with live plot
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Every 100 epochs, plot the decision boundary
    if epoch % 100 == 0:
        plot_decision_boundary(X.numpy(), y.numpy(), model, epoch)

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')

# Create GIF using imageio
images = []
for epoch in range(0, num_epochs, 100):
    img = imageio.imread(f'images/epoch_{epoch}.png')
    images.append(img)

# Save the GIF
imageio.mimsave('xor_training.gif', images, duration=0.5)  # 0.5 seconds per frame

# Clean up saved images
import shutil
shutil.rmtree('images')
