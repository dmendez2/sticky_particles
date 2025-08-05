import numbers
import time

import hoomd
import numpy

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(cpu, seed=1)

# Create a simple cubic configuration of particles
N = 12  # particles per box direction
box_L = 50  # box dimension

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

simulation.state.thermalize_particle_momenta(hoomd.filter.All(), 1.0)

lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
lj.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
lj.r_cut[("A", "A")] = 2.5
integrator = hoomd.md.Integrator(
    methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())], forces=[lj], dt=0.005
)

simulation.operations += integrator


class GaussianVariant(hoomd.variant.Variant):
    def __init__(self, mean, std):
        hoomd.variant.Variant.__init__(self)
        self.mean = mean
        self.std = std

    def __call__(self, timestep):
        return rng.normal(self.mean, self.std)


energy = GaussianVariant(0.1, 0.001)
simulation.run(0)
rng = numpy.random.default_rng(1245)

# we will improve the performance of the InsertEnergyUpdater. 
# Specifically we will change to use the cpu_local_snapshot to update particle velocity. 
# We will use the %%timeit magic function for timing the simulation's run time before and after our optimization. 
# To highlight the difference, we will run the updater every timestep.
class InsertEnergyUpdater(hoomd.custom.Action):
    def __init__(self, energy):
        self._energy = energy

    @property
    def energy(self):
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

energy_action = InsertEnergyUpdater(energy)
energy_operation = hoomd.update.CustomUpdater(action=energy_action, trigger=1)
simulation.operations.updaters.append(energy_operation)

start = time.time()
simulation.run(100)
end = time.time()
print("Time Taken: ", end - start, " Seconds")

# We now show the profile for the optimized code which uses the cpu_local_snapshot for updating velocities.
class InsertEnergyUpdater(hoomd.custom.Action):
    def __init__(self, energy):
        self._energy = energy

    @property
    def energy(self):
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

    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm

    def act(self, timestep):
        part_tag = rng.integers(self._state.N_particles)
        direction = self._get_direction()
        energy = self.energy(timestep)
        with self._state.cpu_local_snapshot as snap:
            # We restrict the computation to the MPI
            # rank containing the particle if applicable.
            # By checking if multiple MPI ranks exist first
            # we can avoid checking the inclusion of a tag id
            # in an array.
            if self._comm.num_ranks <= 1 or part_tag in snap.particles.tag:
                i = snap.particles.rtag[part_tag]
                mass = snap.particles.mass[i]
                magnitude = numpy.sqrt(2 * energy / mass)
                velocity = direction * magnitude
                old_velocity = snap.particles.velocity[i]
                new_velocity = old_velocity + velocity
                snap.particles.velocity[i] = new_velocity

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

# Create and add our modified custom updater
simulation.operations -= energy_operation
energy_action = InsertEnergyUpdater(energy)
energy_operation = hoomd.update.CustomUpdater(action=energy_action, trigger=1)
simulation.operations.updaters.append(energy_operation)

start = time.time()
simulation.run(100)
end = time.time()
print("Time Taken: ", end - start, " Seconds")

# As can be seen, using local snapshot is much faster than calling snapshot each time (Order of magnitude of 10 for this size simulation)
# Snapshot -> O(N)
# Local Snapshot -> O(1)