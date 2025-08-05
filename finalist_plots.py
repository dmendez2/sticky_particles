import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to convert volume fractions to strings which help with directory and file naming
def get_vol_fraction_str(vol_fraction):
    return str(round(vol_fraction*100, 2)).replace('.', '_')

lm_fraction = float(sys.argv[1])
lm_fraction_str = get_vol_fraction_str(lm_fraction)

mxene_fraction = float(sys.argv[2])
mxene_fraction_str = get_vol_fraction_str(mxene_fraction)

finalist_directory = f'data/finalist_permutations/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}'
optimal_permutations = ['experimental', 'multi_double_patch_and_delta_25', 'multi_quadruple_patch_and_delta_30', 'multi_triple_patch_and_delta_30', 'multi_quadruple_patch_and_delta_35', 'multi_triple_patch_and_delta_35', 'multi_double_patch_and_delta_35']
data_tables = []
for p in optimal_permutations:
    data_tables.append(pd.read_csv(f'{finalist_directory}/{p}.csv')['normalized_volume'])
optimal_permutations = ['Experimental', f'2 Patches (\u03B4: 25°)', f'4 Patches (\u03B4: 30°)', f'3 Patches (\u03B4: 30°)', f'4 Patches (\u03B4: 35°)', f'3 Patches (\u03B4: 35°)', f'2 Patches (\u03B4: 35°)']
# Set consistent style
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each category with consistent spacing and styling
for i, (cat, x_vals) in enumerate(zip(optimal_permutations, data_tables)):
    y_vals = np.full_like(x_vals, i) + np.random.normal(0, 0.05, size=len(x_vals))  # Add jitter
    ax.scatter(x_vals, y_vals, s=80, alpha=0.7, edgecolor='k', label=cat)

# Set y-ticks to category labels
ax.set_yticks(np.arange(len(optimal_permutations)))
ax.set_yticklabels(optimal_permutations, fontsize=14)

# Axis labels and title
ax.set_xlabel("Normalized Volume", fontsize=14)
#ax.set_ylabel("Simulation/Experiment", fontsize=14)
#ax.set_title("Normalized Volume Distribution of Large Clusters", fontsize=16)

# Grid and tick styling
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', length=0)  # Clean y-axis

# Optional: add legend or background lines
# ax.legend(title="Group")

plt.tight_layout()
plt.savefig('figures/finalists.png')
