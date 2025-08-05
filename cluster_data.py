import os
import hdbscan
import numpy as np
import matplotlib.pyplot as plt

def Normalize_Data(data):
  # Compute min and max for each axis (x, y, z)
  min_vals = data.min(axis=0)  # shape (3,)
  max_vals = data.max(axis=0)

  # Normalize to [0, 1]
  normalized_data = (data - min_vals) / (max_vals - min_vals)
  return normalized_data

def Get_Data(delta, filename):
  file_path = "data/delta_" + str(delta) + "/" + filename
  if(os.path.exists(file_path)):
    data = np.load(file_path)
    return Normalize_Data(data)
  else: 
    return None

deltas = []
num_clustered = []
for d in range(0, 181):
    positions = Get_Data(d, "mxene_0_25_positions.npy")

    if(positions is not None):
        # Instantiate and fit the HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples = 25)  # tweak as needed
        labels = clusterer.fit_predict(positions)

        labels = labels[labels != -1]
        unique = np.unique(labels)

        if(len(unique) == 1):
            deltas.append(d)
            num_clustered.append(len(labels))
        elif(len(unique) != 0):
            i = 0
            N = 0
            biggest_cluster = 0
            for u in unique:
                size_cluster = len(labels[labels == u])
                if(size_cluster > biggest_cluster):
                    biggest_cluster = size_cluster
                    i = N
                N += 1
            deltas.append(d)
            num_clustered.append(len(labels[labels == unique[i]]))

plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)

fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
ax.scatter(deltas, num_clustered, color='tab:blue', s=40)

ax.set_xlabel("Delta (degrees)", fontsize=14)
ax.set_ylabel("Largest Cluster Size (N particles)", fontsize=14)
ax.set_title("Largest Cluster Size vs Delta", fontsize=16)

ax.tick_params(axis='both', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("data/aggregate_data/cluster_data.png", dpi=300)