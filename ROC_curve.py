import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32)

# split same way as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# define same CNN architecture
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

# load trained model
model = JetCNN().to(device)
model.load_state_dict(torch.load("jet_cnn.pth"))
model.eval()
model.load_state_dict(torch.load("jet_cnn.pth", map_location=device))
# get predictions on test set
test_dataset = TensorDataset(X_test, y_test)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_predictions = []
all_labels      = []

model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        preds   = model(batch_X).squeeze().cpu().numpy()
        all_predictions.extend(preds)
        all_labels.extend(batch_y.numpy())

all_predictions = np.array(all_predictions)
all_labels      = np.array(all_labels)

# compute ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
roc_auc = auc(fpr, tpr)

# plot
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