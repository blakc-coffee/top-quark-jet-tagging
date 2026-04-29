import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── models ──────────────────────────────────────────

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)

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
        return self.fc_layers(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + x)

class JetResNet(nn.Module):
    def __init__(self):
        super(JetResNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResBlock(32)
        self.layer2 = ResBlock(32)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.layer3 = ResBlock(32)
        self.layer4 = ResBlock(32)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.layer5 = ResBlock(32)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)
        x = self.pool3(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

# ── training function ────────────────────────────────

def train_model(model, train_loader, test_loader, epochs=20, name="model"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}")
        
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
        print(f"[{name}] Epoch {epoch+1}/{epochs}  Avg Loss: {avg_loss:.4f}")
    
    # test accuracy
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            preds = (model(batch_X).squeeze() > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total   += len(batch_y)
    print(f"[{name}] Test Accuracy: {correct/total*100:.2f}%")
    
    # get predictions for ROC
    all_preds  = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    
    torch.save(model.state_dict(), f"{name}.pth")
    print(f"[{name}] Saved.")
    
    return np.array(all_preds), np.array(all_labels)

# ── load datasets ────────────────────────────────────

print("\nLoading raw dataset...")
raw  = np.load("dataset_raw.npz")
X_raw = torch.tensor(raw["X"], dtype=torch.float32).unsqueeze(1)
y_raw = torch.tensor(raw["y"], dtype=torch.float32)

print("Loading preprocessed dataset...")
pre  = np.load("dataset.npz")
X_pre = torch.tensor(pre["X"], dtype=torch.float32).unsqueeze(1)
y_pre = torch.tensor(pre["y"], dtype=torch.float32)

# ── split datasets ───────────────────────────────────

def make_loaders(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.to(device)
    X_test  = X_test.to(device)
    y_train = y_train.to(device)
    y_test  = y_test.to(device)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64, shuffle=False)
    return train_loader, test_loader

train_raw, test_raw = make_loaders(X_raw, y_raw)
train_pre, test_pre = make_loaders(X_pre, y_pre)

# ── train all models ─────────────────────────────────

results = {}

print("\n═══ Model 1: SimpleNN (raw) ═══")
m1 = SimpleNN().to(device)
results["SimpleNN (raw)"] = train_model(m1, train_raw, test_raw, name="model1_simplenn_raw")

print("\n═══ Model 2: SimpleNN (preprocessed) ═══")
m2 = SimpleNN().to(device)
results["SimpleNN (preprocessed)"] = train_model(m2, train_pre, test_pre, name="model2_simplenn_pre")

print("\n═══ Model 3: CNN (preprocessed) ═══")
m3 = JetCNN().to(device)
results["CNN (preprocessed)"] = train_model(m3, train_pre, test_pre, name="model3_cnn_pre")

print("\n═══ Model 4: ResNet (preprocessed) ═══")
m4 = JetResNet().to(device)
results["ResNet (preprocessed)"] = train_model(m4, train_pre, test_pre, name="model4_resnet_pre")

# ── plot all ROC curves ──────────────────────────────

colors = ["red", "orange", "blue", "green"]

plt.figure(figsize=(9, 7))

for (name, (preds, labels)), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate (Background kept)")
plt.ylabel("True Positive Rate (Signal caught)")
plt.title("ROC Curve Comparison — All Models")
plt.legend()
plt.grid(True)
plt.savefig("roc_comparison.png", dpi=150)
plt.show()
print("Saved roc_comparison.png")