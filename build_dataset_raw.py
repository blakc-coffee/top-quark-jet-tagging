import pandas as pd
import numpy as np

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
        y = int((phi - phi_min) / (phi_max - phi_min) * GRID_SIZE)
        
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            image[x][y] += pt
    
    return image

X = []
y = []

df_iter = pd.read_hdf("train.h5", key="table", chunksize=100)
count = 0

for df in df_iter:
    for _, row in df.iterrows():
        img   = jet_to_image_raw(row)
        label = row["is_signal_new"]
        
        X.append(img)
        y.append(label)
        
        count += 1
        print(f"Processing jet {count}/500000", end="\r")
        
        if count >= 500000:
            break
    if count >= 500000:
        break

X = np.array(X)
y = np.array(y)

print("\nX shape:", X.shape)
np.savez("dataset_raw.npz", X=X, y=y)
print("Saved dataset_raw.npz")