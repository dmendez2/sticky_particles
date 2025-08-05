import hoomd
import numpy
import matplotlib.pyplot as plt
import h5py
import os
import gsd.hoomd
import datetime

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)
simulation.create_state_from_gsd(
    filename="snapshots/randomized_2.gsd"
)

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
simulation.run(0)

# Many classes in HOOMD-blue provide special properties called loggable quantities. 
# For example, the Simulation class provides timestep, tps, and others. 
# The reference documentation labels each of these as Loggable. 
# You can also examine the loggables property to determine the loggable quantities:
print(simulation.loggables)

# The ThermodynamicQuantities class computes a variety of thermodynamic properties in MD simulations. 
# These are all loggable.
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All()
)
simulation.operations.computes.append(thermodynamic_properties)
print()
print(thermodynamic_properties.loggables)

# Loggable quantities are class properties or methods. 
# You can directly access them in your code.
print(simulation.timestep)
print(thermodynamic_properties.kinetic_temperature)

# Each loggable quantity has a category, which is listed both in the reference documentation and in loggables. 
# The category is a string that identifies the quantity's type or category. Example categories include:
# scalar - numbers
# sequence - arrays of numbers
# string - strings of characters
# particle - arrays of per-particle values

# Add each of the quantities you would like to store to a Logger. 
# The Logger will maintain these quantities in a list and provide them to the Writer when needed.
logger = hoomd.logging.Logger(categories=["scalar", "sequence"])

# You can add loggable quantities from any number of objects to a Logger. 
# Logger uses the namespace of the class to assign a unique name for each quantity. 
# Call add to add all quantities provided by thermodynamic_properties:
logger.add(thermodynamic_properties)

# You can also select specific quantities to add with the quantities argument. 
# Add only the timestep and walltime quantities from Simulation:
logger.add(simulation, quantities=["timestep", "walltime"])

# Some file checking, if the file exists, remove it
if(os.path.exists("log/example.h5")):
    os.remove("log/example.h5")

# Use the HDF5Log writer to store the quantities provided by logger to a HDF5 (.h5) file.
hdf5_writer = hoomd.write.HDF5Log(
    trigger=hoomd.trigger.Periodic(1000), filename="log/example.h5", mode="x", logger=logger
)
simulation.operations.writers.append(hdf5_writer)
simulation.run(100_000)

# Remove the writer from the simulation to close the file:
# Note: This step is not necessary in typical workflows where a simulation script writes a log file and exits before a later analysis script reads the file.
simulation.operations.writers.remove(hdf5_writer)

# Open the logged file
hdf5_file = h5py.File(name="log/example.h5", mode="r")

# Access the potential energy
print(hdf5_file["hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"][:])

timestep = hdf5_file["hoomd-data/Simulation/timestep"][:]
potential_energy = hdf5_file["hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"][:]

# Plot the potential energy
fig = plt.Figure(figsize=(5, 3.09))
ax = fig.add_subplot()
ax.plot(timestep, potential_energy)
ax.set_xlabel("timestep")
ax.set_ylabel("potential energy")
fig.savefig("figures/potential_energy.png")

# HDF5Log writes sequence, particle, bond, etc... quantities in addition to scalers. 
# For example, the pressure tensor is a 6 element array on each frame:
print(hdf5_file["hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor"][0])

# MD forces provide a number of loggable quantities including their contribution to the system energy, 
# but also per-particle energy contributions (in energies) and per-particle forces, torques, and virials.
print(lj.loggables)

# Add the per-particle LJ energies and forces to a logger:
logger = hoomd.logging.Logger()
logger.add(lj, quantities=["energies", "forces"])

# Some file checking, if the file exists, remove it
if(os.path.exists("snapshots/trajectory_logged.gsd")):
    os.remove("snapshots/trajectory_logged.gsd")

# Create the GSD writer to write the simulation trajectory:
gsd_writer = hoomd.write.GSD(
    filename="snapshots/trajectory_logged.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="xb",
    filter=hoomd.filter.All(),
)
simulation.operations.writers.append(gsd_writer)

# Set the logger attribute and GSD will also store the selected quantities.
gsd_writer.logger = logger

# Run the simulation:
simulation.run(100000)

# Close the GSD file so it is readable in the next code cell. 
# In typical workflows, you will run separate simulation and analysis scripts 
# and the file will automatically close when your simulation script exits.
simulation.operations.writers.remove(gsd_writer)
del gsd_writer

# Use the gsd package to open the file:
traj = gsd.hoomd.open("snapshots/trajectory_logged.gsd", mode="r")

# The log data for a specific frame is stored in the log dictionary for that frame.
print(traj[0].log.keys())

# GSD prepends particles/ to the logged name of per-particle quantities. 
# The quantities are NumPy arrays with N_particles elements. Here are a few slices:
traj[-1].log["particles/md/pair/LJ/energies"][0:10]
traj[-1].log["particles/md/pair/LJ/forces"][0:10]

# You can use these arrays as inputs to any computation or plotting tools:
numpy.mean(traj[-1].log["particles/md/pair/LJ/forces"], axis=0)

fig = plt.figure(figsize=(5, 3.09))
ax = fig.add_subplot()
ax.hist(traj[-1].log["particles/md/pair/LJ/energies"], 100)
ax.set_xlabel("potential energy")
ax.set_ylabel("count")
fig.savefig("figures/histogram.png")

# As with scalar quantities, the array quantities are stored separately in each frame. 
# Use a loop to access a range of frames and compute time-series data or averages.

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)
simulation.create_state_from_gsd(
    filename="snapshots/randomized_2.gsd"
)

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

# The Table writer formats log quantities in human-readable text and writes them to stdout or a file. 
# Table only supports scalar and string quantities due to the limitations of this format. 
# This section shows you how to use Table to display status information during a simulation run.

# The categories argument to Logger defines the categories that it will accept.
logger = hoomd.logging.Logger(categories=["scalar", "string"])

# Log the simulation timestep and tps quantities:
logger.add(simulation, quantities=["timestep", "tps"])

# You can also log user-defined quantities using functions, callable class instances, or class properties. 
# For example, this class computes the estimated time remaining:

class Status:
    def __init__(self, simulation):
        self.simulation = simulation

    @property
    def seconds_remaining(self):
        try:
            return (
                self.simulation.final_timestep - self.simulation.timestep
            ) / self.simulation.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))

# Assign the loggable quantity using the tuple (object, property_name, flag), where flag is the string name of the flag for this quantity. 
# (The tuple for callable objects would be (callable, flag)).
status = Status(simulation)
logger[("Status", "etr")] = (status, "etr", "string")

# Represent the namespace of your user-defined quantity with a tuple of strings - ('Status', 'etr') above.
#  You can use any number of arbitrary strings in the tuple to name your quantity.

# Table is a Writer that formats the quantities in a Logger into a human readable table. 
# Create one that triggers periodically:
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=5000), logger=logger)
simulation.operations.writers.append(table)
simulation.run(100000)
simulation.operations.writers.remove(table)

# Table writes to stdout by default. 
# It can write to a file (or any Python file-like object with write and flush methods) instead.
file = open("log/advanced_log_example.txt", mode="x", newline="\n")
table_file = hoomd.write.Table(
    output=file, trigger=hoomd.trigger.Periodic(period=5000), logger=logger
)
simulation.operations.writers.append(table_file)
simulation.run(100000)