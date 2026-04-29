import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 40

# image WITHOUT preprocessing
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
        y = int((phi - phi_min) / (phi_max - phi_min) * GRID_SIZE)
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            image[x][y] += pt
    return image


# image WITH preprocessing
def jet_to_image(jet):
    etas = []
    phis = []
    pts  = []
    
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
        
        etas.append(eta)
        phis.append(phi)
        pts.append(pt)
    
    if len(pts) == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE))
    
    etas = np.array(etas)
    phis = np.array(phis)
    pts  = np.array(pts)
    
    # STEP 1 — CENTER
    eta_center = np.sum(etas * pts) / np.sum(pts)
    phi_center = np.sum(phis * pts) / np.sum(pts)
    etas = etas - eta_center
    phis = phis - phi_center
    
    # STEP 2 — ROTATE
    cov_etaeta = np.sum(pts * etas**2) / np.sum(pts)
    cov_phiphi = np.sum(pts * phis**2) / np.sum(pts)
    cov_etaphi = np.sum(pts * etas * phis) / np.sum(pts)
    angle = 0.5 * np.arctan2(2 * cov_etaphi, cov_etaeta - cov_phiphi)
    
    etas_rot =  etas * np.cos(angle) + phis * np.sin(angle)
    phis_rot = -etas * np.sin(angle) + phis * np.cos(angle)
    etas = etas_rot
    phis = phis_rot
    
    # STEP 3 — FLIP
    if np.sum(pts[etas > 0]) < np.sum(pts[etas < 0]):
        etas = -etas
    if np.sum(pts[phis > 0]) < np.sum(pts[phis < 0]):
        phis = -phis
    
    # build image
    image = np.zeros((GRID_SIZE, GRID_SIZE))
    eta_min, eta_max = -3, 3
    phi_min, phi_max = -np.pi, np.pi
    
    for eta, phi, pt in zip(etas, phis, pts):
        x = int((eta - eta_min) / (eta_max - eta_min) * GRID_SIZE)
        y = int((phi - phi_min) / (phi_max - phi_min) * GRID_SIZE)
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            image[x][y] += pt
    
    return image


# preview preprocessing on one jet
df_iter_preview = pd.read_hdf("train.h5", key="table", chunksize=1)
for df in df_iter_preview:
    sample_jet = df.iloc[0]
    break

img_raw  = jet_to_image_raw(sample_jet)
img_proc = jet_to_image(sample_jet)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_raw,  cmap="hot")
axes[0].set_title("Before Preprocessing")
axes[0].axis("off")
axes[1].imshow(img_proc, cmap="hot")
axes[1].set_title("After Preprocessing")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("preprocessing_comparison.png", dpi=150)
plt.show()
print("Saved preprocessing_comparison.png")

input("Press Enter to continue building dataset...")

# build dataset
X = []
y = []

df_iter = pd.read_hdf("train.h5", key="table", chunksize=100)
count = 0

for df in df_iter:
    for _, row in df.iterrows():
        img   = jet_to_image(row)
        label = row["is_signal_new"]
        
        X.append(img)
        y.append(label)
        
        count += 1
        print(f"Processing jet {count}/100000", end="\r")
        
        if count >= 100000:
            break
    if count >= 100000:
        break

X = np.array(X)
y = np.array(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

np.savez("dataset.npz", X=X, y=y)
print("Saved dataset.npz")