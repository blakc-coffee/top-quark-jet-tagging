import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

print("NumPy:", np.array([1,2,3]))
print("Pandas:", pd.DataFrame({"a":[1,2]}))
print("Torch:", torch.tensor([1,2,3]))

plt.plot([1,2,3],[4,5,6])
plt.show()