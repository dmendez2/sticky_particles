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

def get_patch_number(orientation):
    if orientation == 'multi_double':
        return 2
    elif orientation == 'multi_triple':
        return 3
    elif orientation == 'multi_quadruple':
        return 4

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

# Read in simulation parameters
patch_orientations = ['multi_double', 'multi_triple', 'multi_quadruple']
plotting_names = ['2 Patches', '3 Patches', '4 Patches']
lm_fraction = float(sys.argv[1])
lm_fraction_str = get_vol_fraction_str(lm_fraction)
mxene_fraction = float(sys.argv[2])
mxene_fraction_str = get_vol_fraction_str(mxene_fraction)

# Make directory for finalists
finalist_directory = f'data/finalist_permutations/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
os.makedirs(finalist_directory, exist_ok = True)

# Let's call a particle something that has at least 512 voxels (8 x 8 x 8 Cube)
# This is used in the literature to ensure we are not looking at noise
# It's a rather harsh cut but only changes the max normalized cluster size by 0.01 so I think it should be ok
experimental_data = pd.read_csv('data/experimental/lm_0_25_experimental.csv')
experimental_data = experimental_data[experimental_data['voxel_count'] > 512]
experimental_data['normalized_volume'] = experimental_data['volume']/np.sum(experimental_data['volume'])

# If I order the normalized volumes in order of size then the resulting curve finally begins to rise from ~ 0 to something tangible 
# When the normalized volume is 0.005
filtered_experimental_data = experimental_data[experimental_data['normalized_volume'] > 0.005]

# Create an experimental curve (X-axis is just the ranked order of the cluster size) for later Frechet distance measurement
# E.g. smallest cluster -> 1, next smallest -> 2, .... largest cluster -> N_clusters
exp_curve = np.column_stack((np.arange(len(filtered_experimental_data)), filtered_experimental_data['normalized_volume'].to_numpy()))
filtered_experimental_data.to_csv(f'{finalist_directory}/experimental.csv')

# Go through each patch orientation and each delta
sim_name = []
frechet_data = []
i = 0
for orientation in patch_orientations:
    current_plot_name = plotting_names[i]
    for delta in np.arange(1, 181, 1):
        # Read in the equilibirum trajectory 
        # This would be the highest frame so figure out what the highest frame is
        # We save these frames as csvs in the process_trajectory script
        data_input_directory = f'data/equilibrium_clusters/{orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}/delta_{delta}'
        highest_frame = 0
        if os.path.exists(data_input_directory) and os.path.isdir(data_input_directory):
            highest_frame = get_highest_csv_number(data_input_directory + '')
        data_input_path = f'{data_input_directory}/{highest_frame}.csv'
        if not os.path.exists(data_input_path):
            continue
    
        # Cut the simulation data at a normalized volume of 0.005 
        # This allows us to compare the experimental and simulation data (Everything below this is just a normalized volume of around 0)
        df = pd.read_csv(data_input_path)
        df = df.sort_values(by = "normalized_volume")
        filtered_df = df[df['normalized_volume'] > 0.005]

        # Some of the simulations either stay completely un-clustered so they have no 'lift' in their curve
        # Some of the simulations are nearly completely clustered into 1 cluster
        # We can filter out these bad results by requiring that the sims we look at have at least 25% of the data points above the cutoff
        if(len(filtered_df) > 0.25 * len(filtered_experimental_data)):
            # Save this result as a 'finalist'
            filtered_df.to_csv(f'{finalist_directory}/{orientation}_patch_and_delta_{delta}.csv')

            # Create a simulation curve similar to experimental curve above
            sim_curve = np.column_stack((np.arange(len(filtered_df)), filtered_df['normalized_volume'].to_numpy()))

            # Get the patch number and save the simulation name as a nice reader friendly title
            patch_number = get_patch_number(orientation)
            sim_name.append(f'{patch_number} Patches (\u03B4: {delta}°)')

            # Compute the frechet distance between the experimental and simulation curve
            frechet_data.append(frechet_dist(exp_curve, sim_curve))

# Create an output directory for the frechet data
frechet_output_directory = f'figures/frechet/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
os.makedirs(frechet_output_directory, exist_ok = True)

# Convert to numpy arrays for easy sorting
sim_name = np.array(sim_name)
frechet_data = np.array(frechet_data)

# Sort by Fréchet distance (ascending, best matches first)
sort_indices = np.argsort(frechet_data)
sim_name_sorted = sim_name[sort_indices]
frechet_data_sorted = frechet_data[sort_indices]

# Create a bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(sim_name_sorted, frechet_data_sorted, color='steelblue', edgecolor='black')

# Aesthetics
ax.set_xlabel("Fréchet Distance", fontsize=12)
#ax.set_title("Similarity Between Experimental and Simulated Curves", fontsize=14, weight='bold')
ax.invert_yaxis()  # best matches at the top
ax.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig(f"{frechet_output_directory}/aggregate.png", dpi=300)
plt.show()