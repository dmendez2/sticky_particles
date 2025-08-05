import h5py
import hoomd
import IPython
import numpy

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(cpu, seed=1)

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

# For this section, we will demonstrate writing a custom trajectory writer using h5py. 
# We will start by implementing the ability to store positions, timesteps, and box dimensions in an HDF5 file.
class HDF5Writer(hoomd.custom.Action):
    def __init__(self, filename, mode):
        super().__init__()
        self._filename = filename   # renamed to underscore to mark private
        self._mode = mode
        self._cur_frame = 1
        self._state = None          # initialize here so it's always present
        self._file = None           # renamed and initialized early

        if mode not in {"w", "w-", "x", "a", "r+"}:
            raise ValueError("mode must be writable.")

        try:
            self._file = h5py.File(filename, mode)
            self._write_metadata()
            frames = list(self._file.keys())
            if frames:
                self._cur_frame = max(map(int, frames)) + 1
        except Exception as e:
            print(f"Failed to open HDF5 file {filename!r} in mode {mode!r}: {e}")
            raise

    def attach(self, simulation):
        self._state = simulation.state

    def detach(self):
        if self._file and self._file.id:
            self._file.close()

    def _write_metadata(self):
        """Write the file metadata that defines the type of hdf5 file."""
        if "app" in self._file.attrs:
            if self._file.attrs.app != "hoomd-v3":
                message = 'HDF5 file metadata "app" is not "hoomd-v3".'
                raise RuntimeError(message)
        else:
            self._file.attrs.app = "hoomd-v3"

        if "version" not in self._file.attrs:
            self._file.attrs.version = "1.0"

    def act(self, timestep):
        """Write out a new frame to the trajectory."""
        new_frame = self._file.create_group(str(self._cur_frame))
        self._cur_frame += 1
        positions = new_frame.create_dataset(
            "positions", (self._state.N_particles, 3), dtype="f8"
        )
        snapshot = self._state.get_snapshot()
        positions[:] = snapshot.particles.position
        new_frame.attrs["timestep"] = timestep
        box_array = numpy.concatenate((self._state.box.L, self._state.box.tilts))
        new_frame.attrs["box"] = box_array

    def attach(self, simulation):
        self._state = simulation.state

    def detach(self):
        if hasattr(self, "file") and getattr(self.file, "id", None):
            self.file.close()

    def __del__(self):
        self._file.close()

# Define a function that creates a HDF5Writer wrapped in a custom writer.
# This function will make creating our custom writer easier. 
# We will now add an HPMC sphere integrator and our custom writer to our simulation and run for 1000 steps.
h5_writer = hoomd.write.CustomWriter(action=HDF5Writer("snapshots/traj.h5", "w"), trigger=100)
integrator = hoomd.hpmc.integrate.Sphere()
integrator.shape["A"] = {"diameter": 1.0}

simulation.operations += integrator
simulation.operations += h5_writer
simulation.run(1000)