import itertools
import math
import os
from mpi4py import MPI

import gsd.hoomd
import hoomd
import matplotlib
import numpy as np
from PIL import Image
from skimage.draw import disk
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball
from tqdm import tqdm
import tifffile

def radian_to_degree_conversion(radian):
    return int(radian * 180/np.pi)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("MPI enabled:", hoomd.version.mpi_enabled)

    # Some file checking, if the file exists, remove it
    if(os.path.exists("snapshots/initial.gsd")):
        os.remove("snapshots/initial.gsd")

    m = 12
    N_particles = 2 * m**3
    spacing = 1.2
    K = math.ceil(N_particles ** (1 / 3))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    position = position[0:N_particles]
    orientation = [(1, 0, 0, 0)] * N_particles

    # gsd snapshot
    snapshot = gsd.hoomd.Frame()
    snapshot.particles.N = N_particles
    snapshot.particles.position = position
    snapshot.particles.orientation = orientation
    snapshot.particles.typeid = [0] * N_particles
    snapshot.particles.types = ["A"]
    snapshot.configuration.box = [L, L, L, 0, 0, 0]

    with gsd.hoomd.open(name="snapshots/initial.gsd", mode="x") as f:
        f.append(snapshot)

# Ensure rank 0 finishes writing before others read
comm.Barrier()

radial_step = np.pi/180
deltas = np.arange(np.pi/180, np.pi + radial_step, radial_step)
for d in deltas:
    # build simulation
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=0)
    simulation.create_state_from_gsd(filename="snapshots/initial.gsd")

    # parameters for the angular step potential
    this_delta = d
    this_epsilon = 5.0 
    this_lambda = 1.2
    this_sigma = 1.0
    this_kT = 3.0

    # Add an integrable integrator to simulation
    mc = hoomd.hpmc.integrate.Sphere(kT = this_kT)
    mc.shape["A"] = dict(diameter=this_sigma, orientable=True)
    simulation.operations.integrator = mc

    # We wish to compress the system so that the particles compose a 25% volume fraction
    # Create an initial box object which we get from our current simulation
    initial_box = simulation.state.box

    # Compute the volume of the sphere (Recall, that radius of sphere = sigma)
    V_sphere = 4/3 * math.pi * this_sigma**3

    # We now create a final box which we append with the properties of our target box
    # Namely, we give the final box the final volume we wish to have
    final_box = hoomd.Box.from_box(initial_box)
    final_volume_fraction = 0.25
    final_box.volume = simulation.state.N_particles * V_sphere / final_volume_fraction

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
    step.params[("A", "A")] = dict(epsilon=[-this_epsilon], r=[this_lambda * this_sigma])

    # Mask the isotropic potential with an angular step potential
    angular_step = hoomd.hpmc.pair.AngularStep(isotropic_potential=step)
    angular_step.mask["A"] = dict(directors=[(1.0, 0, 0)], deltas=[this_delta])

    mc.pair_potentials = [angular_step]

    simulation.run(0)

    # Randomize the particle positions (At high KT, the particles do not bond)
    simulation.run(5000)

    # Decrease the temperature now and run the simulation for longer, the particles will form clusters
    mc.kT = 0.1
    simulation.run(50000)
    if rank == 0:
        print("Ending Simulation: Delta Is " + str(radian_to_degree_conversion(d)) + " Degrees")

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

        # === Normalize positions to [0, box_size] if needed ===
        positions = positions % box_size

        # === Prepare output folder ===
        angle_degrees = radian_to_degree_conversion(d)
        output_dir = "data/delta_" + str(angle_degrees) + "/"
        os.makedirs(output_dir, exist_ok=True)

        # Setup to save positions of spherical particles
        grid_shape = (512, 512, 512)
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        # Compute voxel size (real units per voxel)
        voxel_size = [box_size[i] / grid_shape[i] for i in range(3)]  # assumes cubic voxels

        # Convert real radius to voxel units (assumes radius = sigma / 2)
        radius_voxels = int(np.ceil((this_sigma) / voxel_size[0]))

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

        np.save(output_dir + "positions.npy", positions)
        tifffile.imwrite(output_dir + "3D.tif", voxel_grid, dtype=np.uint8)

        # === Parameters ===
        image_size = (512, 512)        # pixels in x and y
        slice_thickness = 1.0          # thickness in z for each cross-section
        pixel_size = box_size[0] / image_size[0]  # microns/pixel or unit/pixel

        # === Compute number of slices ===
        z_min, z_max = 0, box_size[2]
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
            radius_units = this_sigma/2
            radius_pixels = int(np.ceil(radius_units / pixel_size))
            # Draw particles
            for x, y in zip(x_pix, y_pix):
                if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                    rr, cc = disk((y, x), radius=radius_pixels, shape=img.shape)
                    img[rr, cc] = 255

            # Save image
            img_pil = Image.fromarray(img)
            img_pil.save(f"{output_dir}/slice_{i:03d}.tif")

        print("Saving: Delta is " + str(radian_to_degree_conversion(d)) + " Degrees")

        #fill_fraction = np.sum(img > 0) / np.size(img)
        #print(f"Slice {i:03d} fill fraction: {fill_fraction:.2%}")



    #print(f"Saved {n_slices} slices to {output_dir}/")