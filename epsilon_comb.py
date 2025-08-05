import itertools
import math
import os

import gsd.hoomd
import hoomd
import matplotlib
import numpy as np
from PIL import Image
from skimage.draw import disk

def radian_to_degree_conversion(radian):
    return int(radian * 180/np.pi)

def float_to_string(val):
    truncated_val = round(val, 2)
    str_val = str(truncated_val).replace('.', '_')
    return str_val

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
print("Number of Particles: ", N_particles)

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

E_step = 0.25
energies = np.arange(E_step, 10.0 + E_step, E_step)
for E in energies:

    # build simulation
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=0)
    simulation.create_state_from_gsd(filename="snapshots/initial.gsd")

    # parameters for the angular step potential
    this_delta = np.pi/4
    this_epsilon = E
    this_lambda = 1.5
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
    step.params[("A", "A")] = dict(epsilon=[-this_epsilon], r=[this_lambda * this_delta])

    # Mask the isotropic potential with an angular step potential
    angular_step = hoomd.hpmc.pair.AngularStep(isotropic_potential=step)
    angular_step.mask["A"] = dict(directors=[(1.0, 0, 0)], deltas=[this_delta])

    mc.pair_potentials = [angular_step]

    simulation.run(0)

    # Some file checking, if the file exists, remove it
    if(os.path.exists("log/log.gsd")):
        os.remove("log/log.gsd")

    logger = hoomd.logging.Logger()
    logger.add(
        mc,
        quantities=["pair_energy"],
    )
    gsd_writer = hoomd.write.GSD(
        filename="log/log.gsd",
        trigger=hoomd.trigger.Periodic(10),
        mode="xb",
        filter=hoomd.filter.Null(),
        logger=logger,
    )
    simulation.operations.writers.append(gsd_writer)

    # Randomize the particle positions (At high KT, the particles do not bond)
    simulation.run(500)

    # Decrease the temperature now and run the simulation for longer, the particles will form clusters
    mc.kT = 0.1
    simulation.run(10000)

    # Assume you have a simulation object already set up:
    snap = simulation.state.get_snapshot()

    # Positions are a numpy array of shape (N_particles, 3)
    positions = snap.particles.position

    # === Input: positions is (N_particles, 3) ===
    box = simulation.state.box
    box_size = [box.Lx, box.Ly, box.Lz]  # Simulation box dimensions (x, y, z)
    print(box_size)

    # === Parameters ===
    image_size = (512, 512)        # pixels in x and y
    slice_thickness = 1.0          # thickness in z for each cross-section
    pixel_size = box_size[0] / image_size[0]  # microns/pixel or unit/pixel

    # === Prepare output folder ===
    str_energy = float_to_string(E)
    print(str_energy)
    output_dir = "data/epsilon_" + str_energy + "/"
    os.makedirs(output_dir, exist_ok=True)

    # === Compute number of slices ===
    z_min, z_max = 0, box_size[2]
    n_slices = int((z_max - z_min) / slice_thickness)

    # === Normalize positions to [0, box_size] if needed ===
    positions = positions % box_size

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
        radius_units = this_sigma / 2
        radius_pixels = int(np.ceil(radius_units / pixel_size))

        # Draw particles
        for x, y in zip(x_pix, y_pix):
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                rr, cc = disk((y, x), radius=radius_pixels, shape=img.shape)
                img[rr, cc] = 255

        # Save image
        img_pil = Image.fromarray(img)
        img_pil.save(f"{output_dir}/slice_{i:03d}.tif")

        fill_fraction = np.sum(img > 0) / np.size(img)
        print(f"Slice {i:03d} fill fraction: {fill_fraction:.2%}")

    print(f"Saved {n_slices} slices to {output_dir}/")