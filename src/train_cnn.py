import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load dataset
data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

# convert to tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32)

print("X shape:", X.shape)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# move to GPU
X_train = X_train.to(device)
X_test  = X_test.to(device)
y_train = y_train.to(device)
y_test  = y_test.to(device)

# create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# define CNN
class JetCNN(nn.Module):
    def __init__(self):
        super(JetCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 10 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

# move model to GPU
model = JetCNN().to(device)

# loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training loop
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_X, batch_y in progress_bar:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        predictions = model(batch_X).squeeze()
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}  Avg Loss: {avg_loss:.4f}")

# test accuracy
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        predictions = (model(batch_X).squeeze() > 0.5).float()
        correct += (predictions == batch_y).sum().item()
        total   += len(batch_y)

print(f"\nTest Accuracy: {correct/total*100:.2f}%")

# save model
torch.save(model.state_dict(), "jet_cnn.pth")
print("Model saved.")

# ROC curve
all_predictions = []
all_labels      = []

model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        preds   = model(batch_X).squeeze().cpu().numpy()
        all_predictions.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels      = np.array(all_labels)

fpr, tpr, _ = roc_curve(all_labels, all_predictions)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="red", lw=2, label=f"CNN (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate (Background kept)")
plt.ylabel("True Positive Rate (Signal caught)")
plt.title("ROC Curve — Jet Tagger")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print(f"AUC: {roc_auc:.4f}")