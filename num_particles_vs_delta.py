import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# Function to convert volume fractions to strings which help with directory and file naming
def get_vol_fraction_str(vol_fraction):
    return str(round(vol_fraction*100, 2)).replace('.', '_')

# Converts radians to degrees
def radian_to_degree_conversion(radian):
    return math.ceil(radian * 180/np.pi)

# We know the volume fraction of the liquid metal and the radius of each liquid metal particle
# Therefore, we can compute the volume that we need to compress the simulation box towards to ensure the liquid metal volume fraction is satisfied
def calculate_volume_of_box(num_particles, r, lm_fraction):
    vol_sphere = 4/3 * np.pi * r**3
    vol_box = (vol_sphere * num_particles)/lm_fraction
    return vol_box

# Formula to compute the surface area of a spherical sector (This models the area of a patch)
# 2 * PI * r^2 * (1 - cos(delta))
# Multiply by a thin thickness to get a volume (According to Mason Zadan's paper, avg diameter of LM particles was 248 nm)
# According to ACY Yuen (2021), thickness of Mxene sheets are 3 nm
# Therefore I choose a thickness of 3/248 ~ 0.01
def calculate_spherical_sector_surface_volume(delta, r, thickness = 0.005):
    return 2 * np.pi * r**2 * (1 - np.cos(delta)) * thickness

# Formula to compute the number of mxene patches we require
# This depends on the mxene volume fraction, the size of the patches (dependent on delta), and the patches per particle
def calculate_num_patchy_particles(delta, r, num_particles, patches_per_particle, lm_fraction, mxene_fraction, thickness = 0.005):
    mxene_area = patches_per_particle * calculate_spherical_sector_surface_volume(delta, r)
    box_volume = calculate_volume_of_box(num_particles, r, lm_fraction)
    num_patches = int((mxene_fraction * box_volume) / mxene_area)
    return num_patches

# Read in passed arguments for the script
final_lm_fraction = float(sys.argv[1])
final_mxene_fraction = float(sys.argv[2])
lm_fraction_str = get_vol_fraction_str(final_lm_fraction)
mxene_fraction_str = get_vol_fraction_str(final_mxene_fraction)

sigma = 1
radius = sigma/2
m = 18
N_particles = 2 * m**3
plt.figure(figsize=(8, 5))  # Make the canvas larger
plt.style.use("seaborn-v0_8-whitegrid")  # Clean grid style

colors = plt.cm.viridis(np.linspace(0, 1, 6))  # Color map for 6 lines

for i in range(1, 7):
    patchy_data = []
    delta_data = []
    for delta in np.arange(np.pi / 180, np.pi / i + np.pi / 180, np.pi / 180):
        N_patchy = calculate_num_patchy_particles(delta, radius, N_particles, i, final_lm_fraction, final_mxene_fraction)
        patchy_fraction = N_patchy / N_particles
        if patchy_fraction > 1.0:
            continue
        patchy_data.append(patchy_fraction * 100)  # Convert to percent
        delta_data.append(radian_to_degree_conversion(delta))

    plt.plot(delta_data, patchy_data, label=f'{i} patches', linewidth=2, color=colors[i-1])

# Labels and title
plt.xlabel(r'Patch angle $\delta$ (°)', fontsize=14)
plt.ylabel('Patchy Particle Fraction (%)', fontsize=14)
plt.title('Fraction of Patchy Particles vs. Patch Angle', fontsize=16)

# Tick formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Legend
plt.legend(title='Patches Per Particle', fontsize=11, title_fontsize=12)

# Grid and layout
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save at high resolution
plt.savefig('figures/num_patchy_vs_delta.png', dpi=300)