import itertools
import math
import os
import sys
from mpi4py import MPI

import gsd.hoomd
import hoomd
import matplotlib
import numpy as np
from PIL import Image
from collections import Counter
from skimage.draw import disk
from skimage.morphology import ball
from tqdm import tqdm
import tifffile

# Creates a Temperature Ramp Updator which linearly changes the temperature of simulation 
# Starts at an initial temperature, start_kT, and transitions to a final temperature, end_kT, within a given number of steps (ramp_steps)
class TemperatureRamp(hoomd.custom.Action):
    def __init__(self, mc_integrator, start_kT, end_kT, ramp_steps):
        self.mc = mc_integrator
        self.start_kT = start_kT
        self.end_kT = end_kT
        self.ramp_steps = ramp_steps

    def act(self, timestep):
        if timestep < self.ramp_steps:
            progress = timestep / self.ramp_steps
            new_kT = self.start_kT + progress * (self.end_kT - self.start_kT)
            self.mc.kT = new_kT
        else:
            self.mc.kT = self.end_kT

# Class to hold the patch information 
# Each patch orientation has different patches per particle and directors which define the Kern-Frenkel potential
# Currently have support for 1-6 patches per particle
class Patch_Information():
    def __init__(self, patch_orientation):
        self.__patch_orientation = patch_orientation
        self.__patches_per_particle = 0
        self.__is_proper_orientation = False
        self.__directors = None

        # Number of patches is trivial (single = 1, double = 2, etc.)
        # The directors are unit vectors which dictate where the patches are located
        # Single is trivial, just along x-axis 
        # Double is one patch along positive x-axis, one along negative x-axis
        # Triple has the patches laid out along an equilateral triangle in the xy-plane
        # Quadruple has tetrahedral geometry
        # Quintuple has trigonal-bipyramidal geometry
        # Sextuple has octahedral geometry
        if(patch_orientation == 'single'):
            self.__patches_per_particle = 1
            self.__is_proper_orientation = True
            self.__directors = [(1.0, 0, 0)]
        elif(patch_orientation == 'double'):
            self.__patches_per_particle = 2
            self.__is_proper_orientation = True
            self.__directors = [(1.0, 0, 0), (-1.0, 0, 0)] 
        elif(patch_orientation == 'triple'):
            self.__patches_per_particle = 3
            self.__is_proper_orientation = True
            self.__directors = [(1.0, 0, 0), (-1/2, np.sqrt(3)/2, 0), (-1/2, -np.sqrt(3)/2, 0)] 
        elif(patch_orientation == 'quadruple'):
            self.__patches_per_particle = 4
            self.__is_proper_orientation = True
            self.__directors = np.array([(1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)])/np.sqrt(3)
        elif(patch_orientation == 'quintuple'):
            self.__patches_per_particle = 5
            self.__is_proper_orientation = True
            self.__directors = [(1.0, 0, 0), (-1/2, np.sqrt(3)/2, 0), (-1/2, -np.sqrt(3)/2, 0), (0, 0, 1.0), (0, 0, -1.0)] 
        elif(patch_orientation == 'sextuple'):
            self.__patches_per_particle = 6
            self.__is_proper_orientation = True
            self.__directors = [(1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0), (0, -1.0, 0), (0, 0, 1.0), (0, 0, -1.0)]
    
    @property
    def patch_orientation(self):
        return self.__patch_orientation  # Getter

    @property
    def patches_per_particle(self):
        return self.__patches_per_particle  # Getter

    @property
    def is_proper_patch_orientation(self):
        return self.__is_proper_orientation  # Getter

    @property
    def directors(self):
        return self.__directors  # Getter

# Reverse engineered log-normal distribution parameters for Mason's diameter measurements of the liquid metal particles
# Mode = exp[mu - sigma^2] -> log(mode) = mu - sigma^2
# Mean = exp[mu + 0.5 * sigma^2] -> 2*log(mean) = 2*mu + sigma^2
# So I can solve for mu = [2*log(mean) + log(mode)]/3
# Then  for sigma = sqrt[mu - log(Mode)]
# OR sigma = sqrt(2* [log(Mean) - mu]) 
def log_normal_parameters(my_mode = 75, my_mean = 248, normalizing_diameter = 248):
    mode = my_mode / normalizing_diameter
    mean = my_mean / normalizing_diameter

    mu = (2*np.log(mean) + np.log(mode))/3
    sigma = np.sqrt(mu - np.log(mode))
    #sigma = np.sqrt(2 * (np.log(mean) - mu))

    return mu, sigma

# We assume each Mxene patch has the same size
# So the deltas are the same for each patch 
def get_deltas(delta, patches_per_particle):
    return [delta] * patches_per_particle

# Function to convert volume fractions to strings which help with directory and file naming
def get_vol_fraction_str(vol_fraction):
    return str(round(vol_fraction*100, 2)).replace('.', '_')

# Converts radians to degrees
def radian_to_degree_conversion(radian):
    return math.ceil(radian * 180/np.pi)

# We know the volume fraction of the liquid metal and the radius of each liquid metal particle
# Therefore, we can compute the volume that we need to compress the simulation box towards to ensure the liquid metal volume fraction is satisfied
def calculate_volume_of_box(radii, lm_fraction):
    vol_spheres = np.sum(4/3 * np.pi * radii**3)
    vol_box = vol_spheres /lm_fraction
    return vol_box

# Formula to compute the surface area of a spherical sector (This models the area of a patch)
# 2 * PI * r^2 * (1 - cos(delta))
# Multiply by a thin thickness to get a volume
def calculate_spherical_sector_surface_volume(delta, radii, thickness = 0.02):
    return 2 * np.pi * radii**2 * (1 - np.cos(delta)) * thickness

def calculate_delta_from_volume(V, radii, thickness = 0.02):
    arg = 1 - V/(2*np.pi * radii**2 * thickness)
    if arg < -1 or arg > 1:
        return np.pi
    else:
        return np.arccos(1 - V/(2*np.pi * radii**2 * thickness))

def dynamic_delta_assigner(radii_bins, particle_ids, mean_delta, mean = 1):
    idx = np.argmin(np.abs(radii_bins - mean))
    V = calculate_spherical_sector_surface_volume(mean_delta, radii_bins[idx])
    deltas = np.zeros(len(radii_bins))
    for i in range(0, len(deltas)):
        if(i == idx):
            deltas[idx] = mean_delta
        else:
            deltas[i] = calculate_delta_from_volume(V, radii_bins[i])
    return deltas[particle_ids]

def dynamic_patch_generator(box_volume, mxene_fraction, radii, deltas, num_patches, N_particles):
    N_patches = 0
    vol_fraction = 0
    patch_limit = num_patches * N_particles
    patch_map = np.zeros(N_particles)
    while np.abs(vol_fraction - mxene_fraction) > 0.0005 and N_patches < patch_limit:
        i = np.random.randint(0, N_particles)
        if(patch_map[i] < num_patches):
            patch_map[i] += 1
            mxene_volume  = np.sum(patch_map * calculate_spherical_sector_surface_volume(deltas, radii))
            vol_fraction = mxene_volume/box_volume
            N_patches += 1
    print("Mxene Volume Fraction is: ", vol_fraction)
    return patch_map, N_patches

# Formula to compute the number of mxene patches we require
# This depends on the mxene volume fraction, the size of the patches (dependent on delta), and the patches per particle
def calculate_num_patches(mean_delta, radii, radii_bins, particle_ids, num_patches, N_particles, lm_fraction, mxene_fraction, thickness = 0.02):
    box_volume = calculate_volume_of_box(radii, lm_fraction)
    deltas = dynamic_delta_assigner(radii_bins, particle_ids, mean_delta)
    patch_map, num_patches = dynamic_patch_generator(box_volume, mxene_fraction, radii, deltas, num_patches, N_particles)
    return patch_map, num_patches

def place_random_patches(patch_map, N_particles, N_patches, max_patches = 4):
    patches_placed = 0
    while(patches_placed < N_patches):
        i = np.random.randint(0, N_particles)
        if(patch_map[i] < max_patches):
            patch_map[i] += 1
            patches_placed += 1
    return patch_map

def get_patch_orientation_probabilities(patch_map, N_particles, N_patch_orientations = 7):
    counts = Counter(patch_map)
    rng = np.random.default_rng(seed=42)  # Optional: set a seed for reproducibility
    patch_fractions = np.zeros(N_patch_orientations)
    for i in range(0, N_patch_orientations):
        if i in counts:
            patch_fractions[i] = counts[i]/N_particles
    type_ids = rng.choice([0, 1, 2, 3, 4, 5, 6], size = N_particles, p = patch_fractions)
    return type_ids

def get_particle_types(patch_map):
    counts = Counter(patch_map)
    types = []
    for c in counts:
        if c == 0:
            types.append('normal')
        elif c == 1:
            types.append('single')
        elif c == 2:
            types.append('double')
        elif c == 3:
            types.append('triple')
        elif c == 4:
            types.append('quadruple')
        elif c == 5:
            types.append('quintuple')
        elif  c == 6:
            types.append('sextuple')
    print(counts)
    print(types)
    return types

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set seed for reproducibility
np.random.seed(42)

# Read in passed arguments for the script
final_lm_fraction = float(sys.argv[1])
final_mxene_fraction = float(sys.argv[2])
max_patches = int(sys.argv[3])
patch_orientation = 'multi'
lm_fraction_str = get_vol_fraction_str(final_lm_fraction)
mxene_fraction_str = get_vol_fraction_str(final_mxene_fraction)

# Fixed variables for the Kern-Frenkel model
average_sigma = 1.0
this_lambda = 1.012
this_epsilon = 5.0

# Create directories for saving data later if they do not exist
snapshot_directory = None
trajectory_directory = None
data_directory = None
log_directory = None
if(rank == 0):
    snapshot_directory = f"snapshots/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}"
    trajectory_directory = f"trajectories/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}"
    data_directory = f"data/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}"
    log_directory = f"log/energies/{patch_orientation}_patch/lm_{lm_fraction_str}/mxene_{mxene_fraction_str}"

    os.makedirs(snapshot_directory, exist_ok = True)
    os.makedirs(trajectory_directory, exist_ok = True)
    os.makedirs(data_directory, exist_ok = True)
    os.makedirs(log_directory, exist_ok = True)

    print("Final LM Fraction: ", final_lm_fraction)
    print("Final MXene Fraction: ", final_mxene_fraction)
    print("Saving Results To File: ", "mxene_" + mxene_fraction_str)

# Broadcast directories to all ranks
snapshot_directory = comm.bcast(snapshot_directory, root = 0)
trajectory_directory = comm.bcast(trajectory_directory, root = 0)
data_directory = comm.bcast(data_directory, root = 0)

# Create the delta angles (in radians) for the Kern-Frenkel model
degree_step = 5
max_delta = 180/max_patches
deltas = np.arange(5, max_delta + degree_step, degree_step)
for delta in deltas:
    d = np.deg2rad(delta)
    angle_degrees = int(delta)
    trajectory_path = None
    initial_snapshot_file_path = None
    skip_delta = None
    types = None
    if rank == 0:
        print("MPI enabled:", hoomd.version.mpi_enabled)
        print("Delta is: ", angle_degrees, " degrees")

        # Create a uniformly spaced lattice of the sticky particles, spaced at least sigma*lambda away from each other
        m = 18
        N_particles = 2 * m**3
        spacing = average_sigma * 1.2
        K = math.ceil(N_particles ** (1 / 3))
        L = K * spacing
        x = np.linspace(-L / 2, L / 2, K, endpoint=False)
        position = list(itertools.product(x, repeat=3))
        position = position[0:N_particles]
        orientation = [(1, 0, 0, 0)] * N_particles

        # Sample particle sizes from exponential distribution
        mu, var = log_normal_parameters()
        rng = np.random.default_rng(seed=42)
        sigma_distribution = rng.lognormal(mean = mu, sigma = var, size = N_particles)
        while np.any(sigma_distribution > 7):
            # Resample just the outliers
            n_resample = np.sum(sigma_distribution > 7)
            sigma_distribution[sigma_distribution > 7] = rng.lognormal(mean=mu, sigma=var, size=n_resample)
        radial_distribution = sigma_distribution / 2

        # For each diameter of particle, I need to have a new integrator
        # If I were to do this for every particle, I would wait exceptionally long for even one simulation run to finish
        # Therefore, I will discretize the diameter distribution into bins (similar to a histogram)
        num_bins = 10
        min_val = np.min(sigma_distribution[sigma_distribution > 0])  # avoid log(0)
        max_val = np.max(sigma_distribution)
        bin_edges = np.linspace(np.min(sigma_distribution), np.max(sigma_distribution), num_bins + 1)
        diameter_ids = np.digitize(sigma_distribution, bin_edges[1:-1])
        print(Counter(diameter_ids))

        # Now assign each particle to its bin based on what diameter it sampled from the exponential distribution
        diameter_bins = np.array([sigma_distribution[diameter_ids == i].mean() for i in range(num_bins)])
        radii_bins = diameter_bins/2
        print(Counter(diameter_bins))

        # Create list of diameters from the diameter bins and ids
        diameters = diameter_bins[diameter_ids]
        radii = diameters / 2

        # Compute the max number of patchy particles possible
        patch_map, N_patchy = calculate_num_patches(d, radii, radii_bins, diameter_ids, max_patches, N_particles, final_lm_fraction, final_mxene_fraction)
        patchy_fraction = N_patchy/N_particles
        print("Patch Fraction Is: ", patchy_fraction)
        skip_delta = (patchy_fraction >= 1) 

    # Broadcast the skip delta boolean to all ranks
    # If necessary, skip this delta value
    skip_delta = comm.bcast(skip_delta, root=0)
    if(skip_delta):
        continue

    if rank == 0:
        print("Delta Is: ", angle_degrees)
        print("Patch Fraction Is: ", patchy_fraction)

        # Assign particle types randomly, ensuring that we maintain the fraction of patchy to normal particles we computed earlier
        type_ids = get_patch_orientation_probabilities(patch_map, N_particles)
        types = get_particle_types(patch_map)

        print(f"Non-patchy: {np.sum(type_ids == 0)}")
        print(f"Patchy: {np.sum(type_ids != 0)}")
        print(f"Total particles: {N_particles}")

        # gsd snapshot (Saves all information of our particles to a snapshot)
        snapshot = gsd.hoomd.Frame()
        snapshot.particles.N = N_particles
        snapshot.particles.position = position
        snapshot.particles.orientation = orientation
        snapshot.particles.typeid = type_ids
        snapshot.particles.types = types
        snapshot.configuration.box = [L, L, L, 0, 0, 0]

        # Initial snapshot file path
        initial_snapshot_file_path = f"{snapshot_directory}/delta_{str(angle_degrees)}.gsd"

        # Some file checking, if the file exists, remove it
        if(os.path.exists(initial_snapshot_file_path)):
            os.remove(initial_snapshot_file_path)

        # Save the snapshot
        with gsd.hoomd.open(name=initial_snapshot_file_path, mode="x") as f:
            f.append(snapshot)

        # Trajectory path
        trajectory_path = f"{trajectory_directory}/delta_{str(angle_degrees)}.gsd"

        # Some file checking, if the file exists, remove it
        if(os.path.exists(trajectory_path)):
            os.remove(trajectory_path)

    # Ensure rank 0 finishes writing before others read
    comm.Barrier()

    # Broadcast trajectory and initial snapshot paths to all ranks
    trajectory_path = comm.bcast(trajectory_path, root = 0)
    initial_snapshot_file_path = comm.bcast(initial_snapshot_file_path, root = 0)
    types = comm.bcast(types, root = 0)

    # build simulation
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=42)
    simulation.create_state_from_gsd(filename=initial_snapshot_file_path)

    # parameters for the angular step potential
    this_kT = 5.0

    # Add an integrable integrator to simulation
    mc = hoomd.hpmc.integrate.Sphere(kT = this_kT)
    mc.shape[types] = dict(diameter=this_sigma, orientable=True)
    simulation.operations.integrator = mc

    # We wish to compress the system so that the particles compose a 25% volume fraction
    # Create an initial box object which we get from our current simulation
    initial_box = simulation.state.box

    # We now create a final box which we append with the properties of our target box
    # Namely, we give the final box the final volume we wish to have
    final_box = hoomd.Box.from_box(initial_box)
    final_box.volume = calculate_volume_of_box(radii, final_lm_fraction)

    # Create a compress updator which periodically compresses the system until we get our desired volume fraction
    # Between these periodic compressions, the monte carlo algorithm ensures that the particles are translated/rotated so there is no overlap
    # Add the updator to the simulation
    compress = hoomd.hpmc.update.QuickCompress(
        trigger=hoomd.trigger.Periodic(10), target_box=final_box
    )
    simulation.operations.updaters.append(compress)

    # Create another periodic trigger for the tuner
    # The tuner can change the monte carlo guesses for variables
    # Here, we tune the guesses for the variables (d, a) which control the translation and rotation steps
    # As the system compresses, the space becomes smaller and larger values for (d,a) which used to be accepted are no longer optimal
    # We therefore use the tuner so it can make the guesses smaller over time and ensure the acceptance ratio does not become too low
    # Here, our target acceptance ratios are 0.2
    # This greatly affects performance
    periodic = hoomd.trigger.Periodic(10)
    tune = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["a", "d"],
        target=0.2,
        trigger=periodic,
        max_translation_move=0.2,
        max_rotation_move=0.2,
    )
    simulation.operations.tuners.append(tune)

    # We run the system until the compression of the system finishes (Or in rare cases of non-convergence, when a very large timescale passes)
    while not compress.complete and simulation.timestep < 1e6:
        simulation.run(1000)

    # If the compression failed, we print out a failure message
    if not compress.complete:
        message = "Compression failed to complete"
        raise RuntimeError(message)

    # Remove compressor and tuner
    simulation.operations.updaters.remove(compress)
    simulation.operations.tuners.remove(tune)

    # U = U_isotropic * U_angular
    step = hoomd.hpmc.pair.Step()
    for i in range(0, len(types)):
        for j in range(i, len(types)):
            type_i = types[i]
            type_j = types[j]
            if type_i == 'normal' or type_j == 'normal':
                step.params[(type_i, type_j)] = dict(epsilon=[0.0], r=[0.0001])
            else:
                step.params[(type_i, type_j)] = dict(epsilon=[-this_epsilon], r=[this_lambda * this_sigma])

    # Mask the isotropic potential with an angular step potential
    angular_step = hoomd.hpmc.pair.AngularStep(isotropic_potential=step)
    for t in types:
        if t == 'normal':
            angular_step.mask['normal'] = dict(directors=[(1.0, 0, 0)], deltas=[np.pi])
        else:
            # Get patch information from the Patch Information Class
            myPatchInformation = Patch_Information(t)
            # Get deltas for our patch orientation
            these_deltas = get_deltas(d, myPatchInformation.patches_per_particle)
            # Apply directors and deltas for this patch orientation
            angular_step.mask[t] = dict(directors = myPatchInformation.directors, deltas = these_deltas)

    # Apply angular potential to particles
    mc.pair_potentials = [angular_step]

    # Add a writer to keep track of trajectory
    gsd_writer = hoomd.write.GSD(filename = trajectory_path,
                                trigger=hoomd.trigger.Periodic(1000),
                                mode='wb',
                                filter=hoomd.filter.All())
    simulation.operations.writers.append(gsd_writer)

    simulation.run(0)

    # Randomize the particle positions (At high KT, the particles do not bond)
    simulation.run(1000)

    # Path for the logging_file
    logging_file_path = f"{log_directory}/delta_{str(angle_degrees)}.gsd"

    # Some file checking, if the file exists, remove it
    if(os.path.exists(logging_file_path)):
        os.remove(logging_file_path)

    # Add a logger to keep track of the potential energy of the system
    logger = hoomd.logging.Logger()
    logger.add(
        mc,
        quantities = ["pair_energy"],
    )
    gsd_writer = hoomd.write.GSD(
        filename=logging_file_path,
        trigger=hoomd.trigger.Periodic(1000),
        mode="xb",
        filter=hoomd.filter.Null(),
        logger=logger,
    )
    simulation.operations.writers.append(gsd_writer)

    # Decrease the temperature now and run the simulation for longer, the particles will form clusters
    # Create a temperature updater to run every 100 steps which changes the temperature from start_kT to end_kT over the total number of steps
    temp_ramp = hoomd.update.CustomUpdater(
        action=TemperatureRamp(mc, start_kT=5.0, end_kT=0.1, ramp_steps=50000),
        trigger=hoomd.trigger.Periodic(100)
    )
    simulation.operations.updaters.append(temp_ramp)
    simulation.run(2000000)
    if rank == 0:
        print("Ending Simulation: Delta Is " + str(angle_degrees) + " Degrees")

    # Assume you have a simulation object already set up:
    # This line must be done with all ranks. 
    # Each rank did one part of the Domain Decomposition
    # We are now reintegrating all the domains together
    snap = simulation.state.get_snapshot()

    if rank == 0:

        # Positions are a numpy array of shape (N_particles, 3)
        positions = snap.particles.position

        # === Input: positions is (N_particles, 3) ===
        box = simulation.state.box
        box_size = [box.Lx, box.Ly, box.Lz]  # Simulation box dimensions (x, y, z)
        print("Length of Box: ", box_size)

        # === Normalize positions to [0, box_size] if needed ===
        positions = positions % box_size

        # Setup to save positions of spherical particles
        grid_shape = (512, 512, 512)
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        # Compute voxel size (real units per voxel)
        voxel_size = [box_size[i] / grid_shape[i] for i in range(3)]  # assumes cubic voxels

        # Convert real radius to voxel units (assumes radius = sigma / 2)
        radius_voxels = int(np.ceil((this_sigma/2) / voxel_size[0]))

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
            valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < np.array(grid_shape)),axis=1)
            voxel_coords = voxel_coords[valid_mask]

            # Set the voxels to 1 (or 255)
            voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 255

        np.save(f"{data_directory}/delta_{angle_degrees}.npy", positions)
        tifffile.imwrite(f"{data_directory}/delta_{angle_degrees}.tif", voxel_grid, dtype=np.uint8)