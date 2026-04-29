import pandas as pd

# load one jet
df_iter = pd.read_hdf("train.h5", key="table", chunksize=1)

for df in df_iter:
    jet = df.iloc[0]
    break

# convert to table
particles = []

for i in range(200):
    E = jet[f"E_{i}"]
    px = jet[f"PX_{i}"]
    py = jet[f"PY_{i}"]
    pz = jet[f"PZ_{i}"]

    if E == 0:
        continue

    particles.append([i, E, px, py, pz])

particle_df = pd.DataFrame(particles, columns=["id", "E", "PX", "PY", "PZ"])

print(particle_df.head(10))
print("\nTotal particles:", len(particle_df))