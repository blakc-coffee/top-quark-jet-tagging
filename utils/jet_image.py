import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load 1 sample
df_iter = pd.read_hdf("train.h5", key="table", chunksize=1)

for df in df_iter:
    jet = df.iloc[0]
    break

etas = []
phis = []
pts = []

# loop over particles
for i in range(200):
    E = jet[f"E_{i}"]
    px = jet[f"PX_{i}"]
    py = jet[f"PY_{i}"]
    pz = jet[f"PZ_{i}"]

    if E == 0:
        continue

    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)

    # avoid division issues
    if (E - pz) == 0:
        continue

    eta = 0.5 * np.log((E + pz) / (E - pz))

    etas.append(eta)
    phis.append(phi)
    pts.append(pt)

# plot
plt.scatter(etas, phis, s=np.array(pts)*0.1)
plt.xlabel("eta")
plt.ylabel("phi")
plt.title("Jet visualization")
plt.show()
