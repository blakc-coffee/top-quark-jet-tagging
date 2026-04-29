import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load dataset
data = np.load("dataset.npz")
X = data["X"]
y = data["y"]

print(f"Total jets: {len(X)}")
print(f"Signal jets: {int(y.sum())}")
print(f"Background jets: {int((y==0).sum())}")

# plot 10 jets — 5 signal, 5 background
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

signal_indices     = np.where(y == 1)[0][:5]
background_indices = np.where(y == 0)[0][:5]

# plot 10 jets — 5 signal, 5 background
for i, ax in enumerate(axes[0]):
    ax.imshow(X[signal_indices[i]], cmap="hot")
    ax.set_title("Signal")
    ax.axis("off")

for i, ax in enumerate(axes[1]):
    ax.imshow(X[background_indices[i]], cmap="hot")
    ax.set_title("Background")
    ax.axis("off")

plt.suptitle("5 Signal (top) vs 5 Background (bottom)", fontsize=16)
plt.tight_layout()
plt.savefig("10_jets.png", dpi=150)
plt.show()
print("Saved 10_jets.png")
# average overlap

signal_imgs     = X[y == 1]
background_imgs = X[y == 0]

diff = np.mean(signal_imgs, axis=0) - np.mean(background_imgs, axis=0)

# before vs after preprocessing comparison
GRID_SIZE = 40

def jet_to_image_raw(jet):
    image = np.zeros((GRID_SIZE, GRID_SIZE))
    eta_min, eta_max = -3, 3
    phi_min, phi_max = -np.pi, np.pi
    for i in range(200):
        E  = jet[f"E_{i}"]
        px = jet[f"PX_{i}"]
        py = jet[f"PY_{i}"]
        pz = jet[f"PZ_{i}"]
        if E == 0:
            continue
        pt  = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        if (E - pz) == 0:
            continue
        eta = 0.5 * np.log((E + pz) / (E - pz))
        x = int((eta - eta_min) / (eta_max - eta_min) * GRID_SIZE)
        y_idx = int((phi - phi_min) / (phi_max - phi_min) * GRID_SIZE)
        if 0 <= x < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
            image[x][y_idx] += pt
    return image

# load 500 raw jets
df_iter = pd.read_hdf("train.h5", key="table", chunksize=500)
for df in df_iter:
    break

raw_signal     = []
raw_background = []

for _, row in df.iterrows():
    img = jet_to_image_raw(row)
    if row["is_signal_new"] == 1:
        raw_signal.append(img)
    else:
        raw_background.append(img)

# before vs after in one plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0][0].imshow(np.mean(raw_signal, axis=0), cmap="hot")
axes[0][0].set_title("Before — Average Signal")
axes[0][0].axis("off")

axes[0][1].imshow(np.mean(raw_background, axis=0), cmap="hot")
axes[0][1].set_title("Before — Average Background")
axes[0][1].axis("off")

diff_raw = np.mean(raw_signal, axis=0) - np.mean(raw_background, axis=0)
axes[0][2].imshow(diff_raw, cmap="bwr")
axes[0][2].set_title("Before — Difference")
axes[0][2].axis("off")

axes[1][0].imshow(np.mean(signal_imgs, axis=0), cmap="hot")
axes[1][0].set_title("After — Average Signal")
axes[1][0].axis("off")

axes[1][1].imshow(np.mean(background_imgs, axis=0), cmap="hot")
axes[1][1].set_title("After — Average Background")
axes[1][1].axis("off")

axes[1][2].imshow(diff, cmap="bwr")
axes[1][2].set_title("After — Difference")
axes[1][2].axis("off")

plt.suptitle("Preprocessing Comparison", fontsize=16)
plt.tight_layout()
plt.savefig("preprocessing_before_after.png", dpi=150)
plt.show()
print("Saved preprocessing_before_after.png")