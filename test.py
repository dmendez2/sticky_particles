import hoomd
import gsd.hoomd
import math

simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=42)
simulation.create_state_from_gsd("snapshots/initial.gsd")

mc = hoomd.hpmc.integrate.Sphere(kT=3.0)
mc.shape["A"] = dict(diameter=1.0, orientable=True)
simulation.operations.integrator = mc

# Optional: define a pair potential
step = hoomd.hpmc.pair.Step()
step.params[("A", "A")] = dict(epsilon=[-1.0], r=[1.2])

angular = hoomd.hpmc.pair.AngularStep(isotropic_potential=step)
angular.mask["A"] = dict(directors=[(1.0, 0, 0)], deltas=[math.pi / 4])

mc.pair_potentials = [angular]

simulation.run(0)  # Don't use .run(0)!