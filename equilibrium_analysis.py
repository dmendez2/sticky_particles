import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from similaritymeasures import frechet_dist
from similaritymeasures import area_between_two_curves

# Function to convert volume fractions to strings which help with directory and file naming
def get_vol_fraction_str(vol_fraction):
    return str(round(vol_fraction*100, 2)).replace('.', '_')

def get_highest_csv_number(directory):
    max_number = -1
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            match = re.match(r"(\d+)\.csv", filename)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
    return max_number

def Plot_Histogram(data, delta, bins):
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=bins, kde=False, color="royalblue", edgecolor='black')

    plt.title("Volume Distribution of LM Clusters", fontsize=16)
    plt.xlabel("Volume", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_directory}/{delta}.png", dpi=300)
    plt.close()

def Plot_Scatter(data, delta, output_directory):
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
    fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
    ax.scatter(np.arange(0, len(data)), data, color='tab:blue', s=40)
    #ax.axhline(y = np.mean(data), color='red', linestyle='--', linewidth=2, label=f'y = Mean Cluster Volume')

    ax.set_xlabel("Cluster", fontsize=14)
    ax.set_ylabel("Normalized Cluster Volume", fontsize=14)
    ax.set_title("Cluster Volume Distribution", fontsize=16)

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_directory}/{delta}.png", dpi=300)
    plt.close()

# Read in simulation parameters
patch_orientation = sys.argv[1]
lm_fraction = float(sys.argv[2])
lm_fraction_str = get_vol_fraction_str(lm_fraction)
mxene_fraction = float(sys.argv[3])
mxene_fraction_str = get_vol_fraction_str(mxene_fraction)

output_directory = f'figures/equilibrium_volume_analysis/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
os.makedirs(output_directory, exist_ok = True)

experimental_data = pd.read_csv('data/experimental/lm_0_25_experimental.csv')
experimental_data = experimental_data[experimental_data['voxel_count'] > 512]
experimental_data['normalized_volume'] = experimental_data['volume']/np.sum(experimental_data['volume'])
filtered_experimental_data = experimental_data[experimental_data['normalized_volume'] > 0.005]
exp_curve = np.column_stack((np.arange(len(filtered_experimental_data)), filtered_experimental_data['normalized_volume'].to_numpy()))

Plot_Scatter(experimental_data['normalized_volume'], 'experimental', output_directory)
Plot_Scatter(filtered_experimental_data['normalized_volume'], 'filtered_experimental', output_directory)
Plot_Histogram(filtered_experimental_data['normalized_volume'], 'histogram_experimental', int(np.sqrt(len(filtered_experimental_data))))

delta_values = []
frechet_data = []
for delta in np.arange(1, 181, 1):
    data_input_directory = f'data/equilibrium_clusters/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}/delta_{delta}'
    highest_frame = 0
    if os.path.exists(data_input_directory) and os.path.isdir(data_input_directory):
        highest_frame = get_highest_csv_number(data_input_directory + '')
    data_input_path = f'{data_input_directory}/{highest_frame}.csv'
    if not os.path.exists(data_input_path):
        continue
    
    df = pd.read_csv(data_input_path)
    df = df.sort_values(by = "normalized_volume")
    filtered_df = df[df['normalized_volume'] > 0.005]
    Plot_Scatter(df['normalized_volume'], f'delta_{delta}', output_directory)
    Plot_Scatter(filtered_df['normalized_volume'], f'filtered_delta_{delta}', output_directory)
    Plot_Histogram(filtered_df['normalized_volume'], f'histogram_delta_{delta}', int(np.sqrt(len(filtered_df))))

    if(len(filtered_df) > 0.35 * len(filtered_experimental_data)):
        delta_values.append(delta)
        sim_curve = np.column_stack((np.arange(len(filtered_df)), filtered_df['normalized_volume'].to_numpy()))
        frechet_data.append(frechet_dist(exp_curve, sim_curve))

frechet_output_directory = f'figures/frechet/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
os.makedirs(frechet_output_directory, exist_ok = True)

# Plot system energy
plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style (newer seaborn version)
fig, ax = plt.subplots(figsize=(8, 5))  # Bigger, cleaner canvas
ax.plot(delta_values, frechet_data, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=5)
ax.set_xlabel("Delta (Degrees)", fontsize=14)
ax.set_ylabel("Frechet Distance", fontsize=14)
ax.set_title(f'Cluster Volume Similarity Between Experiment and Simulations', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(f'{frechet_output_directory}/{patch_orientation}_patch.png', dpi=300)
plt.close()