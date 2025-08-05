import os
import hoomd
import matplotlib.pyplot as plt
import numpy
import math
import itertools
import gsd.hoomd

# Create a molecular dynamics integrator with a time step of 0.005
# Potentials/ forces/ etc. are evaluated at increments of this time step
integrator = hoomd.md.Integrator(dt=0.005)
print(integrator.dt)

# Print out the list of forces attached to the current system. Currently there are none
print(integrator.forces[:])

# This is the Lennard-Jones potential
sigma = 1
epsilon = 1
r = numpy.linspace(0.95, 3, 500)
V_lj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

# Plot the Lennard-Jones potential
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
ax.plot(r, V_lj)
ax.set_xlabel("r")
ax.set_ylabel("V")
fig.savefig("figures/lj.png")

# Create a neighbor list
# Neighboring particles with r < r_cut would have their pair potentials computed
# Neighboring particles with r > r_cut would not
# Discontinuity at r = r_cut
# In this case, our neighbor list is created through cells
cell = hoomd.md.nlist.Cell(buffer=0.4)

# Add the Lennard-Jones potential to the cell
lj = hoomd.md.pair.LJ(nlist=cell)

# We define interactions between each type of particle
# In this case we only have particle, "A", and define A-A interactions
lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
lj.r_cut[("A", "A")] = 2.5

# Add the force to the integrator for the simulation
integrator.forces.append(lj)

# Constant volume models Newtons Law
# The thermostat scales the velocities to sample the canonical ensemble
#kT is the temperature multiplied by Boltzmann's constant and has units of energy. 
# tau is a time constant that controls the amount of coupling between the thermostat and particle's degrees of freedom. 
# filter is a particle filter object that selects which particles this integration method applies to.
nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
)
integrator.methods.append(nvt)

# Initialize some particles
m = 4
N_particles = 4 * m**3

# Lay out the particles in a lattice
spacing = 1.3
K = math.ceil(N_particles ** (1 / 3))
L = K * spacing
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))

frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position[0:N_particles]
frame.particles.typeid = [0] * N_particles
frame.configuration.box = [L, L, L, 0, 0, 0]
frame.particles.types = ["A"]

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/lattice_2.gsd")):
    os.remove("snapshots/lattice_2.gsd")

# Write particle to a system 
with gsd.hoomd.open(name="snapshots/lattice_2.gsd", mode="x") as f:
    f.append(frame)

# Set up the simulation to run on cpu
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)
simulation.create_state_from_gsd(filename="snapshots/lattice_2.gsd")

# Initiate the integrator
integrator = hoomd.md.Integrator(dt=0.005)

# Initiate the cells
cell = hoomd.md.nlist.Cell(buffer=0.4)

# Connect the cells to the Lennard-Jones potential 
lj = hoomd.md.pair.LJ(nlist=cell)

# Edit the Lennard_Jones potential constants
lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
lj.r_cut[("A", "A")] = 2.5

# Add the potential to the integrator
integrator.forces.append(lj)

# Create the conditions for the simulation and add to the integrator
nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
)
integrator.methods.append(nvt)

# Add the integrator to the simulation
simulation.operations.integrator = integrator

# Notice that the initial velocities are all set to 0
# When using the ConstantVolume or ConstantPressure method with a thermostat, you must specify non-zero initial velocities. 
# The thermostat modifies particle velocities by a scale factor so it cannot scale a zero velocity to a non-zero one.
snapshot = simulation.state.get_snapshot()
print(snapshot.particles.velocity[0:5])

# he thermalize_particle_momenta method will assign Gaussian distributed velocities consistent with the canonical ensemble. 
# It also sets the velocity of the center of mass to 0
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

# You can inspect the snapshot to see the changes that thermalize_particle_momenta produced. 
# Use the ThermodynamicQuantities class to compute properties of the system:
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())

#ThermodynamicQuantities is a Compute, an Operation that computes properties of the system state. 
# Some computations can only be performed during or after a simulation run has started. 
# Add the compute to the operations list and call run(0) to make all properties available without changing the system state:
simulation.operations.computes.append(thermodynamic_properties)
simulation.run(0)

#There are 3N - 3 degrees of freedom in the system. 
# The ConstantVolume integration method conserves linear momentum, so the - 3 accounts for the effectively pinned center of mass.
print(thermodynamic_properties.degrees_of_freedom)

# Following the equipartition theorem, the average kinetic energy of the system should be approximately 1/2 * KT * N_DOF
# Recall that KT for us is set to 1.5
print(1 / 2 * 1.5 * thermodynamic_properties.degrees_of_freedom)

# However, note the discrepancy
print(thermodynamic_properties.kinetic_energy)

# The reason is the temperature does not remain fixed
# The temperature fluctuates around the set value
# Using the current temperature, the average temperature is the same as the equipartition theorem
print(thermodynamic_properties.kinetic_temperature)
print(1/2 * thermodynamic_properties.kinetic_temperature * thermodynamic_properties.degrees_of_freedom)

# Run the simulation
simulation.run(10000)

# Notice that the average kinetic energy has changed again
# It will always remain close to the expected average kinetic energy for the input KT of 1.5
print(thermodynamic_properties.kinetic_energy)

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/randomized_2.gsd")):
    os.remove("snapshots/randomized_2.gsd")

# Save the randomized simulation state
hoomd.write.GSD.write(state=simulation.state, filename="snapshots/randomized_2.gsd", mode="xb")

# Initialize a new simulation
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)
simulation.create_state_from_gsd(filename="snapshots/randomized_2.gsd")

# Set up the cells, potential, and integrator again
integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
lj.r_cut[("A", "A")] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
)
integrator.methods.append(nvt)
simulation.operations.integrator = integrator

# A Variant is a scalar-valued function of the simulation timestep that allows you change simulation parameters in a controlled manner. 
# The Ramp variant defines a function that linearly interpolates from one value to another over a given number of steps from t_start to t_start + t_ramp.
# You could use a Ramp variant to slowly compress a system to a target pressure or anneal to a target temperature.
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=simulation.timestep, t_ramp=20_000)

# Plot the ramp
steps = range(0, 40000, 20)
y = [ramp(step) for step in steps]

fig = plt.figure(figsize=(5, 3.09))
ax = fig.add_subplot()
ax.plot(steps, y)
ax.set_xlabel("timestep")
ax.set_ylabel("ramp")
ax.tick_params("x", labelrotation=30)
fig.savefig("figures/ramp.png")

# To compress the box to a target density, you need to use a BoxVariant that defines all 6 parameters of the simulation box as a function of time.
# InverseVolumeRamp represents a linear ramp in the box's inverse volume (or density when N is constant) from the initial box 
# to the target volume between t_start and t_start + t_ramp while maintaining constant aspect ratios and tilts.

# Previously, we set this density
rho = simulation.state.N_particles / simulation.state.box.volume
print(rho)

final_rho = 1.2
final_volume = simulation.state.N_particles / final_rho

# Create an InverseVolumeRamp that interpolates the initial simulation box to the target volume:
inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
    initial_box=simulation.state.box,
    final_volume=final_volume,
    t_start=simulation.timestep,
    t_ramp=20_000,
)

# Plot what the ramp looks like
steps = range(0, 40000, 20)
y = [inverse_volume_ramp(step)[0] for step in steps]

fig = plt.figure(figsize=(5, 3.09))
ax = fig.add_subplot()
ax.plot(steps, y)
ax.set_xlabel("timestep")
ax.set_ylabel("L")
ax.tick_params("x", labelrotation=30)
fig.savefig("figures/inverse_volume_ramp.png")

# The BoxResize Updater scales the positions of all particles and sets the simulation box following the given BoxVariant.
# Construct a BoxResize updater and add it to the simulation.
box_resize = hoomd.update.BoxResize(
    trigger=hoomd.trigger.Periodic(10),
    box=inverse_volume_ramp,
)
simulation.operations.updaters.append(box_resize)

# Run the simulation to compress the box
simulation.run(20001)

# Confirm the updated density and remove the box resizer from the simulation
print(simulation.state.N_particles / simulation.state.box.volume)
simulation.operations.updaters.remove(box_resize)

# Now let us equilibriate the system
simulation.run(5e5)

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/ordered_fcc.gsd")):
    os.remove("snapshots/ordered_fcc.gsd")

# Save the randomized simulation state
hoomd.write.GSD.write(state=simulation.state, filename="snapshots/ordered_fcc.gsd", mode="xb")