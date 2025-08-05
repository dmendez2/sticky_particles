import os
import sys
import shutil
import gsd.hoomd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Function to convert volume fractions to strings which help with directory and file naming
def get_vol_fraction_str(vol_fraction):
    return str(round(vol_fraction*100, 2)).replace('.', '_')

def radian_to_degree_conversion(radian):
    return math.ceil(radian * 180 / np.pi)

def compute_radius_of_gyration(positions):
    if len(positions) == 0:
        return 0.0
    com = np.mean(positions, axis=0)
    return np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))

def compute_gr_fast(positions, box_size, r_max, dr):
    N = len(positions)
    density = N / (box_size ** 3)

    bins = np.arange(0, r_max + dr, dr)
    r = 0.5 * (bins[1:] + bins[:-1])

    # Query all pairs within r_max
    tree = cKDTree(positions, boxsize=box_size)
    pairs = tree.query_pairs(r_max)
    
    # Compute pairwise distances (accounting for PBC)
    deltas = positions[np.array([i for i, j in pairs])] - positions[np.array([j for i, j in pairs])]
    deltas = (deltas + box_size / 2) % box_size - box_size / 2
    distances = np.linalg.norm(deltas, axis=1)

    # Histogram
    counts, _ = np.histogram(distances, bins)

    # Normalize
    shell_volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    ideal_counts = density * shell_volumes * N
    gr = counts / ideal_counts

    return r, gr

def find_clusters_fast(N, pairs):
    """
    N: number of particles
    pairs: list or set of (i, j) index pairs
    """
    if len(pairs) == 0:
        return np.arange(N), np.ones(N, dtype=int)  # Each particle in its own cluster

    row, col = zip(*pairs)
    data = np.ones(len(row), dtype=int)
    adjacency = coo_matrix((data, (row, col)), shape=(N, N))
    adjacency = adjacency + adjacency.T  # Make symmetric

    n_clusters, labels = connected_components(adjacency)
    return labels, np.bincount(labels)


patch_orientation = sys.argv[1]
lm_fraction = float(sys.argv[2])
mxene_fraction = float(sys.argv[3])
lm_fraction_str = get_vol_fraction_str(lm_fraction)
mxene_fraction_str = get_vol_fraction_str(mxene_fraction)
cutoff = 1.012
sigma = 1
r = sigma/2

for delta in np.arange(25, 26, 1):#(1, 180 + 1, 1):
    print(f"Delta_{delta}")
    trajectory_path = f"trajectories/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}/delta_{delta}.gsd"
    if not os.path.exists(trajectory_path):
        continue

    data_output_directory = f'data/equilibrium_clusters/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}/delta_{delta}'
    if os.path.exists(data_output_directory) and os.path.isdir(data_output_directory):
        shutil.rmtree(data_output_directory)
    
    os.makedirs(data_output_directory, exist_ok=True)
    traj = gsd.hoomd.open(trajectory_path, 'r')
    print(len(traj))

    num_frames = len(traj)
    cluster_sizes = []
    max_cluster_sizes = []
    num_clusters = []
    gyration_radius = []
    #r = []
    #g_r = []

    this_frame = 0
    for frame in traj:
        #if this_frame < 200:
        #    this_frame += 1
        #    continue
        pos = frame.particles.position
        box_L = frame.configuration.box[0]  # assume cubic box
        pos = (pos + box_L / 2.0) % box_L   # shift from [-L/2, L/2) to [0, L)
        tree = cKDTree(pos, boxsize=box_L)  # periodic KD-tree
        pairs = tree.query_pairs(r=cutoff)
        labels, counts = find_clusters_fast(len(pos), pairs)

        # Store Rg for largest cluster
        if len(counts) > 0:
            largest_cluster_index = np.argmax(counts)  # ID of largest cluster
            mask = labels == largest_cluster_index
            largest_cluster_pos = pos[mask]
            gyration_radius.append(compute_radius_of_gyration(largest_cluster_pos))
        else:
            gyration_radius.append(0.0)

        mean_cluster_size = 0
        if len(counts[counts > 5]) != 0:
            mean_cluster_size = np.mean(counts[counts > 5]) 
        cluster_sizes.append(mean_cluster_size)
        num_clusters.append(len(counts[counts > 5]))
        max_cluster_sizes.append(np.max(counts))

        # get g(r)
        #radius, radial_distribution = compute_gr_fast(pos, box_L, box_L/2, 0.05)
        #r.append(radius)
        #g_r.append(radial_distribution)

        if(this_frame > int(num_frames*0.75)):
            df = pd.DataFrame()
            #df.index = labels
            df['num_particles'] = counts
            df['volume'] = df['num_particles']*4/3*np.pi*r**3
            df['normalized_volume'] = df['volume'] / np.sum(df['volume'])
            df.to_csv(f'{data_output_directory}/{this_frame}.csv')
        this_frame += 1

    # Create figures output directory
    cluster_size_output_directory = f'figures/cluster_size/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
    max_cluster_size_output_directory = f'figures/max_cluster_size/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
    num_clusters_output_directory = f'figures/num_clusters/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
    gyration_radius_output_directory = f'figures/gyration_radius/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
    #g_r_output_directory = f'figures/radial_distribution/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
    os.makedirs(cluster_size_output_directory, exist_ok=True)
    os.makedirs(max_cluster_size_output_directory, exist_ok = True)
    os.makedirs(num_clusters_output_directory, exist_ok=True)
    os.makedirs(gyration_radius_output_directory, exist_ok = True)
    #os.makedirs(g_r_output_directory, exist_ok=True)

    # Plot cluster size
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
    fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
    ax.plot(cluster_sizes, linestyle='-', color='tab:blue', linewidth=2)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_ylabel("Mean Cluster Size", fontsize=14)
    ax.set_title(f'Cluster growth (delta={delta}°)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{cluster_size_output_directory}/delta_{delta}.png', dpi=300)
    plt.close()

    # Plot max cluster size
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
    fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
    ax.plot(max_cluster_sizes, linestyle='-', color='tab:blue', linewidth=2)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_ylabel("Max Cluster Size", fontsize=14)
    ax.set_title(f'Max Cluster Size (delta={delta}°)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{max_cluster_size_output_directory}/delta_{delta}.png', dpi=300)
    plt.close()

    # Plot num clusters
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
    fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
    ax.plot(num_clusters, linestyle='-', color='tab:blue', linewidth=2)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_ylabel("Number of Clusters", fontsize=14)
    ax.set_title(f'Cluster Agglomeration (delta={delta}°)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{num_clusters_output_directory}/delta_{delta}.png', dpi=300)
    plt.close()

    # Plot radius of gyration
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
    fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
    ax.plot(gyration_radius, linestyle='-', color='tab:blue', linewidth=2)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_ylabel("Radius of Gyration", fontsize=14)
    ax.set_title(f'Largest Cluster Stability (delta={delta}°)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{gyration_radius_output_directory}/delta_{delta}.png', dpi=300)
    plt.close()

    # Plot radius of gyration
    #plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
    #fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
    #ax.plot(gyration_radius, linestyle='-', color='tab:blue', linewidth=2)
    #ax.set_xlabel("Frame", fontsize=14)
    #ax.set_ylabel("g(r)", fontsize=14)
    #ax.set_title(f'Radial Distribution of Clusters (delta={delta}°)', fontsize=16)
    #ax.tick_params(axis='both', labelsize=12)
    #ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    #plt.tight_layout()
    #plt.savefig(f'{g_r_output_directory}/delta_{delta}.png', dpi=300)
    #plt.close()