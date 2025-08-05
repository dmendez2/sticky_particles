import numpy as np
from PIL import Image
from skimage.draw import disk
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball
import tifffile

m = 18
N_particles = 2 * m**3
L = 58
sigma = 1.0  # Diameter
min_dist = sigma
positions = []

max_attempts = 100000
attempts = 0

while len(positions) < N_particles and attempts < max_attempts:
    # Propose a new position
    new_pos = np.random.uniform(-L + sigma, L - sigma, size=3)

    # Check for overlap using KDTree
    if positions:
        tree = cKDTree(positions)
        if tree.query_ball_point(new_pos, r=min_dist):
            attempts += 1
            continue
    
    # Accept the position
    positions.append(new_pos)
    attempts = 0  # reset counter after a successful placement

positions = np.array(positions)

# === Normalize positions to [0, box_size] if needed ===
positions = positions % L

# Setup to save positions of spherical particles
grid_shape = (512, 512, 512)
voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

# Physical box size
L = 38.69
sigma = 1.0

# Compute voxel size (real units per voxel)
voxel_size = [L / grid_shape[i] for i in range(3)]  # assumes cubic voxels

# Convert real radius to voxel units (assumes radius = sigma / 2)
radius_voxels = int(np.ceil((sigma) / voxel_size[0]))

# Generate a sphere mask
sphere_mask = ball(radius_voxels)  # shape: (2r+1, 2r+1, 2r+1)
sphere_indices = np.nonzero(sphere_mask)  # tuple of 3 arrays

# Offsets relative to the center
offsets = np.array(sphere_indices).T - radius_voxels  # shape: (N_voxels_in_sphere, 3)

for pos in tqdm(positions):
    # Convert position to voxel index
    center_idx = np.floor(pos / voxel_size).astype(int)  # (z, y, x)

    # Get absolute voxel coordinates for this particle
    voxel_coords = offsets + center_idx  # shape: (N_voxels_in_sphere, 3)

    # Filter out coordinates that are outside bounds
    valid_mask = np.all(
        (voxel_coords >= 0) & (voxel_coords < np.array(grid_shape)),
        axis=1
    )
    voxel_coords = voxel_coords[valid_mask]

    # Set the voxels to 1 (or 255)
    voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 255

np.save("data/uniform_dist/positions.npy", positions)
np.save("data/uniform_dist/voxel_grid_with_radius.npy", voxel_grid)
tifffile.imwrite("data/uniform_dist/3D.tif", voxel_grid, dtype=np.uint8)

# === Parameters ===
image_size = (512, 512)        # pixels in x and y
slice_thickness = 1.0          # thickness in z for each cross-section
pixel_size = L / image_size[0]  # microns/pixel or unit/pixel

# === Compute number of slices ===
z_min, z_max = 0, L
n_slices = int((z_max - z_min) / slice_thickness)

 # === Loop through z-slices ===
for i in range(n_slices):
    z_low = z_min + i * slice_thickness
    z_high = z_low + slice_thickness

    # Select particles in this z-range
    in_slice = positions[(positions[:, 2] >= z_low) & (positions[:, 2] < z_high)]

    # Convert to pixel coordinates
    x_pix = (in_slice[:, 0] / pixel_size).astype(int)
    y_pix = (in_slice[:, 1] / pixel_size).astype(int)

    # Create blank image
    img = np.zeros(image_size, dtype=np.uint8)

    # Create the radius of my pixels based on the radius of the particles
    radius_units = sigma/2
    radius_pixels = int(np.ceil(radius_units / pixel_size))
    # Draw particles
    for x, y in zip(x_pix, y_pix):
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            rr, cc = disk((y, x), radius=radius_pixels, shape=img.shape)
            img[rr, cc] = 255

    # Save image
    img_pil = Image.fromarray(img)
    img_pil.save(f"data/uniform_dist/slice_{i:03d}.tif")