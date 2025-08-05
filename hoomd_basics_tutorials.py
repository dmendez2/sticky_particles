import itertools
import math
import os

import hoomd
import gsd.hoomd
import numpy as np

# First, bind a device to a simulation object. 
# Create a seed for the simulation so we get the same result each time
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed = 1)

# This is a hard particle monte carlo simulation (hpmc)
# The Convex Polyhedron integrator is the vehicle for the simulation
mc = hoomd.hpmc.integrate.ConvexPolyhedron()

# Create an octahedron with a map of its vertices
mc.shape["octahedron"] = dict(
    vertices=[
        (-0.5, 0, 0),
        (0.5, 0, 0),
        (0, -0.5, 0),
        (0, 0.5, 0),
        (0, 0, -0.5),
        (0, 0, 0.5),
    ]
)

# The monte carlo chooses steps that are possible (Don't clip into other particles)
# It can translate up to a specified distance (d)
# It can rotate up to a certain angle (a)
mc.nselect = 2
mc.d["octahedron"] = 0.15
mc.a["octahedron"] = 0.2

# Assign the integrator to the simulation
simulation.operations.integrator = mc

# Determine the number of particles
# Ocahedrons self-assemble into a BCC lattice so this is why we choose the following number of particles
m = 4
N_particles = 2 * m**3

# The octahedrons are created in spheres of radius 1
# We therefore have a spacing of at least 1.2 between particles
# The particles are arranged along Kx, Ky, Kz. Assuming these are equal to, Ciel(K = N_particles ** (1/3))
# The length of our simulation box is then K * Spacing
spacing = 1.2
K = math.ceil(N_particles ** (1 / 3))
L = K * spacing

# Create equally spaced particles in the box
# Only take up to N_Particles (Since K * K * K >= N_particles)
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
position = position[0:N_particles]

# We also must set an orientation 
# The quaternion (1, 0, 0, 0) represents no rotation
orientation = [(1, 0, 0, 0)] * N_particles

# The frame object stores the current state of the system
frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position
frame.particles.orientation = orientation

# Each particle also has a type. 
# In this example, all particles are octahedrons
frame.particles.typeid = [0] * N_particles
frame.particles.types = ["octahedron"]

# A simulation box can be described by length, width, height and 3 tilt factors
frame.configuration.box = [L, L, L, 0, 0, 0]

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/lattice.gsd")):
    os.remove("snapshots/lattice.gsd")

# Write the initial state snapshot to a file, "lattice.gs"
with gsd.hoomd.open(name="snapshots/lattice.gsd", mode="x") as f:
    f.append(frame)

# The simulation can then be initialized from the gsd file
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=12)

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape["octahedron"] = dict(
    vertices=[
        (-0.5, 0, 0),
        (0.5, 0, 0),
        (0, -0.5, 0),
        (0, 0.5, 0),
        (0, 0, -0.5),
        (0, 0, 0.5),
    ]
)

simulation.operations.integrator = mc
simulation.create_state_from_gsd(filename="snapshots/lattice.gsd")

# Create a snapshot of the initial setup
initial_snapshot = simulation.state.get_snapshot()

# Run the system for 10,000 steps with Monte Carlo
simulation.run(10e3)

# Print results of Monte Carlo translations and rotations 
print("Translation Moves: ", mc.translate_moves)
print("Rotation Moves: ", mc.rotate_moves)
print()
print("Accepted Translation Ratio: ", mc.translate_moves[0]/ sum(mc.translate_moves))
print("Accepted Rotation Ratio: ", mc.rotate_moves[0]/ sum(mc.rotate_moves))
print()

# Print Number of Overlaps (Overlaps are bad, and mean that 2 particles are in the same space at once)
print("Number of Overlaps: ", mc.overlaps)  

# Get the snapshot of the final system setup
final_snapshot = simulation.state.get_snapshot()

# Initial Particles Positions
print(initial_snapshot.particles.position[0:4])
print()

# Final Particles Positions
print(final_snapshot.particles.position[0:4])
print()

# Initial Particles Orientations
print(initial_snapshot.particles.orientation[0:4])
print()

# Final Particles Orientations
print(final_snapshot.particles.orientation[0:4])
print()

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/random.gsd")):
    os.remove("snapshots/random.gsd")

# Save the final randomized simulation state
hoomd.write.GSD.write(state=simulation.state, mode="xb", filename="snapshots/random.gsd")

# Computing the volume of our octahedron
# The volume for a regular octahedron is (1/3) * root(2) * a**3
# Where a is the edge length of the octahedron
a = math.sqrt(2) / 2
V_particle = 1 / 3 * math.sqrt(2) * a**3

# Compute the volume fraction [(Num particles * volume of particles)/(volume of box)]
initial_volume_fraction = (
    simulation.state.N_particles * V_particle / simulation.state.box.volume
)
print(initial_volume_fraction)

# We wish to compress the system so that the particles compose a 57% volume fraction
# Create an initial box object which we get from our current simulation
initial_box = simulation.state.box

# We now create a final box which we append with the properties of our target box
# Namely, we give the final box the final volume we wish to have
final_box = hoomd.Box.from_box(initial_box)
final_volume_fraction = 0.57
final_box.volume = simulation.state.N_particles * V_particle / final_volume_fraction

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

# Time it took for the system to compress
print(simulation.timestep, "microseconds")

# The MoveSize tuner should have adjusted the move size to relatively small values by the end of the simulation
print("Translation Step Size: ", mc.d["octahedron"])
print("Rotation Step Size: ", mc.a["octahedron"])

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/compressed.gsd")):
    os.remove("snapshots/compressed.gsd")

# Save the compressed setup
hoomd.write.GSD.write(state=simulation.state, mode="xb", filename="snapshots/compressed.gsd")

# Reset the simulation and load the compressed snapshot
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=12)
mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape["octahedron"] = dict(
    vertices=[
        (-0.5, 0, 0),
        (0.5, 0, 0),
        (0, -0.5, 0),
        (0, 0.5, 0),
        (0, 0, -0.5),
        (0, 0, 0.5),
    ]
)
simulation.operations.integrator = mc
simulation.create_state_from_gsd(filename="snapshots/compressed.gsd")

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/trajectory.gsd")):
    os.remove("snapshots/trajectory.gsd")

# Create a gsd writer object which will periodically write down the state of the system
gsd_writer = hoomd.write.GSD(
    filename="snapshots/trajectory.gsd", trigger=hoomd.trigger.Periodic(1000), mode="xb"
)
simulation.operations.writers.append(gsd_writer)

# We are now interested in equilibrium
# Add a tuner to the simulation to tune the translation and rotation moves
# When studying equilibrium, we can tune the step sizes near the beginning, but not near the end
# Otherwise, our results would not be true equilbrium and would have been affected by the tuning process
tune = hoomd.hpmc.tune.MoveSize.scale_solver(
    moves=["a", "d"],
    target=0.2,
    # We run a hoomd AND trigger
    # The AND trigger runs when all its child processes return True
    # The PERIODIC trigger returns true every t time steps
    # The BEFORE trigger returns true for any time before a max time value, T
    # Therefore, past the time value, T, the tuner will no longer be triggered
    trigger=hoomd.trigger.And(
        [hoomd.trigger.Periodic(100), hoomd.trigger.Before(simulation.timestep + 5000)]
    ),
)
simulation.operations.tuners.append(tune)

# Run the simulation for the 5000 steps necessary for the Before Trigger to no longer go off
simulation.run(5000)

# Now check the acceptance ratios
simulation.run(100)

rotate_moves = mc.rotate_moves
print("Accepted Rotations: ", mc.rotate_moves[0] / sum(mc.rotate_moves))

translate_moves = mc.translate_moves
print("Accepted Translations: ", mc.translate_moves[0] / sum(mc.translate_moves))

# Now we run the simulation to achieve equilbirum
simulation.run(2e5)

simulation.operations.writers.remove(gsd_writer)
del gsd_writer