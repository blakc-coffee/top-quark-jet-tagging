import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# load dataset
data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

# convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# flatten each 40x40 image into 1600 numbers
X = X.view(X.shape[0], -1)

print("X shape after flatten:", X.shape)
print("y shape:", y.shape)

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# define model
model = nn.Sequential(
    nn.Linear(1600, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 50

for epoch in range(epochs):
    predictions = model(X_train).squeeze()
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

# accuracy on training data
with torch.no_grad():
    train_preds = (model(X_train).squeeze() > 0.5).float()
    train_acc = (train_preds == y_train).float().mean()
    print(f"\nTraining Accuracy: {train_acc.item()*100:.2f}%")

# accuracy on test data
with torch.no_grad():
    test_preds = (model(X_test).squeeze() > 0.5).float()
    test_acc = (test_preds == y_test).float().mean()
    print(f"Test Accuracy:     {test_acc.item()*100:.2f}%")