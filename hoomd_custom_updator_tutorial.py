import numbers

import hoomd
import numpy

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)

# Create a simple cubic configuration of particles
N = 5  # particles per box direction
box_L = 20  # box dimension

snap = hoomd.Snapshot(cpu.communicator)
snap.configuration.box = [box_L] * 3 + [0, 0, 0]
snap.particles.N = N**3
x, y, z = numpy.meshgrid(
    *(numpy.linspace(-box_L / 2, box_L / 2, N, endpoint=False),) * 3
)
positions = numpy.array((x.ravel(), y.ravel(), z.ravel())).T
snap.particles.position[:] = positions
snap.particles.types = ["A"]
snap.particles.typeid[:] = 0

simulation.create_state_from_snapshot(snap)
rng = numpy.random.default_rng(1245)

# In this section, we will show how to create a custom updater that modifies the system state. 
# To show this, we will create a custom updater that adds a prescribed amount of energy to a single particle simulating the bombardment of radioactive material into our system. 
# For this problem, we pick a random particle and modify its velocity according to the radiation energy in a random direction.
class InsertEnergyUpdater(hoomd.custom.Action):
    def __init__(self, energy):
        self.energy = energy

    def act(self, timestep):
        snap = self._state.get_snapshot()
        if snap.communicator.rank == 0:
            particle_i = rng.integers(snap.particles.N)
            mass = snap.particles.mass[particle_i]
            direction = self._get_direction()
            magnitude = numpy.sqrt(2 * self.energy / mass)
            velocity = direction * magnitude
            old_velocity = snap.particles.velocity[particle_i]
            new_velocity = old_velocity + velocity
            snap.particles.velocity[particle_i] = new_velocity
        self._state.set_snapshot(snap)

    @staticmethod
    def _get_direction():
        theta, z = rng.random(2)
        theta *= 2 * numpy.pi
        z = 2 * (z - 0.5)
        return numpy.array(
            [
                numpy.sqrt(1 - (z * z)) * numpy.cos(theta),
                numpy.sqrt(1 - (z * z)) * numpy.sin(theta),
                z,
            ]
        )

# We will now use our custom updater with an NVE integrator. Particles will interact via a Lennard-Jones potential. 
# Using the Table writer and a hoomd.logging.Logger, we will monitor the energy, which should be increasing as we are adding energy to the system. 
# We will also thermalize our system to a kT == 1.
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
lj.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
lj.r_cut[("A", "A")] = 2.5
integrator = hoomd.md.Integrator(
    methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())], forces=[lj], dt=0.005
)

thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())
logger = hoomd.logging.Logger(categories=["scalar"])
logger.add(thermo, ["kinetic_energy", "potential_energy"])
logger["total_energy"] = (
    lambda: thermo.kinetic_energy + thermo.potential_energy,
    "scalar",
)

table = hoomd.write.Table(100, logger, max_header_len=1)

simulation.operations += integrator
simulation.operations += thermo
simulation.operations += table

# Create and add our custom updater
energy_operation = hoomd.update.CustomUpdater(
    action=InsertEnergyUpdater(10.0), trigger=100
)
simulation.operations += energy_operation
simulation.run(1000)

# Maybe we want to allow for the energy to be from a distribution. 
# HOOMD-blue has a concept called a variant which allows for quantities that vary over time. 
# Let's change the InsertEnergyupdater to use variants and create a custom variant that grabs a random number from a Gaussian distribution.
class InsertEnergyUpdater(hoomd.custom.Action):
    def __init__(self, energy):
        self._energy = energy

    @property
    def energy(self):
        """A `hoomd.variant.Variant` object."""
        return self._energy

    @energy.setter
    def energy(self, new_energy):
        if isinstance(new_energy, numbers.Number):
            self._energy = hoomd.variant.Constant(new_energy)
        elif isinstance(new_energy, hoomd.variant.Variant):
            self._energy = new_energy
        else:
            message = "energy must be a variant or real number."
            raise ValueError(message)

    def act(self, timestep):
        snap = self._state.get_snapshot()
        if snap.communicator.rank == 0:
            particle_i = rng.integers(snap.particles.N)
            mass = snap.particles.mass[particle_i]
            direction = self._get_direction()
            magnitude = numpy.sqrt(2 * self.energy(timestep) / mass)
            velocity = direction * magnitude
            old_velocity = snap.particles.velocity[particle_i]
            new_velocity = old_velocity + velocity
            snap.particles.velocity[particle_i] = new_velocity
        self._state.set_snapshot(snap)

    @staticmethod
    def _get_direction():
        theta, z = rng.random(2)
        theta *= 2 * numpy.pi
        z = 2 * (z - 0.5)
        return numpy.array(
            [
                numpy.sqrt(1 - (z * z)) * numpy.cos(theta),
                numpy.sqrt(1 - (z * z)) * numpy.sin(theta),
                z,
            ]
        )


class GaussianVariant(hoomd.variant.Variant):
    def __init__(self, mean, std):
        hoomd.variant.Variant.__init__(self)
        self.mean = mean
        self.std = std

    def __call__(self, timestep):
        return rng.normal(self.mean, self.std)

# The Gaussian Variant just chooses values from a Gaussian distribution
energy = GaussianVariant(mean=10.0, std=2.0)
sample_energies = numpy.array([energy(0) for _ in range(1000)])
print(f"Mean: {sample_energies.mean()}, std. dev. {sample_energies.std()}")

# We now use the updated InsertEnergyUpdater in the simulation.
simulation.operations.updaters.remove(energy_operation)
# Create and add our custom updater
energy_operation = hoomd.update.CustomUpdater(
    action=InsertEnergyUpdater(energy), trigger=100
)
simulation.operations.updaters.append(energy_operation)
simulation.run(1000)

# We could continue to improve upon this updater and the execution of this operation. 
# However, this suffices to showcase the ability of non-trivial updaters to affect the simulation state.