import os
import numpy as np
import matplotlib.pyplot as plt

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def Normalize_Data(data):
  # Compute min and max for each axis (x, y, z)
  min_vals = data.min(axis=0)  # shape (3,)
  max_vals = data.max(axis=0)

  # Normalize to [0, 1]
  normalized_data = (data - min_vals) / (max_vals - min_vals)
  return normalized_data

def Get_Uniform_Distribution():
    data = np.load("data/uniform_dist/positions.npy")
    return Normalize_Data(data)

def Get_Data(delta, filename):
  file_path = "data/delta_" + str(delta) + "/" + filename
  if(os.path.exists(file_path)):
    data = np.load(file_path)
    return Normalize_Data(data)
  else: 
    return None

def Plot_KLD(input_filename, output_filename):
  q_x = Get_Uniform_Distribution()
  deltas = []
  klds = []
  for d in range(1, 181):
    p_x = Get_Data(d, input_filename)
    if(p_x is not None):
      deltas.append(d)
      this_kld = KLdivergence(p_x, q_x)
      if(this_kld < 0):
        this_kld = 0
      klds.append(this_kld)

  deltas = np.array(deltas)
  klds = np.array(klds)

  plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)

  fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
  ax.plot(deltas, klds, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=5)

  ax.set_xlabel("Delta (degrees)", fontsize=14)
  ax.set_ylabel("KL Divergence", fontsize=14)
  ax.set_title("KL Divergence vs Delta", fontsize=16)

  ax.tick_params(axis='both', labelsize=12)
  ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

  plt.tight_layout()
  plt.savefig("data/aggregate_data/" + output_filename, dpi=300)

Plot_KLD("positions.npy", "all_patchy_klds.png")
Plot_KLD("mxene_0_25_positions.npy", "mxene_0_25_klds.png")