import gsd.hoomd
import hoomd
import os

# Define the simulation
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=2)
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
simulation.run(0)

# The type_shapes loggable quantity is a representation of the particle shape for each type following the type_shapes specification for the GSD file format. 
# In HPMC simulations, the integrator provides type_shapes:
print(mc.loggables)
print()
print(mc.type_shapes)
print()

# Add the type_shapes quantity to a Logger.
logger = hoomd.logging.Logger()
logger.add(mc, quantities=["type_shapes"])

# Write the simulation trajectory to a GSD file along with the logged quantities:
gsd_writer = hoomd.write.GSD(
    filename="mc_trajectory_logged.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="xb",
    filter=hoomd.filter.All(),
    logger=logger,
)
simulation.operations.writers.append(gsd_writer)

# Run the simulation:
simulation.run(20000)

# Close the gsd file to analyze the data
simulation.operations.writers.remove(gsd_writer)
del gsd_writer

# You can access the shape from scripts using the gsd package:
traj = gsd.hoomd.open("trajectory.gsd", mode="r")

# type_shapes is a special quantity available via particles.type_shapes rather than the log dictionary:
traj[0].particles.type_shapes

# Open the file in OVITO and it will read the shape definition and render particles appropriately.