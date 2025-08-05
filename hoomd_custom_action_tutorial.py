import hoomd

# We create a single particle snapshot for initializing a simulation's state further down.
snap = hoomd.Snapshot()
snap.particles.N = 1
snap.particles.position[:] = [0, 0, 0]
snap.particles.types = ["A"]
snap.particles.typeid[:] = [0]
snap.configuration.box = [10, 10, 10, 0, 0, 0]

# Create a custom action as a subclass of hoomd.custom.Action. Here we will create an action that prints the timestep to standard out.
# The logic of the action goes inside the act method. 
# All actions must define this function, and it must take in the simulation timestep; 
# this is passed in when the action is called in the HOOMD-blue run loop.
class PrintTimestep(hoomd.custom.Action):
    def act(self, timestep):
        print(timestep)

# Let's go ahead and create a PrintTimestep object.
custom_action = PrintTimestep()

# o let an Operations object know what kind of action our custom action is, we must wrap it in a subclass of hoomd.custom.CustomOperation. 
# We have three options as discussed in the previous section: an updater, writer, or tuner. 
# Since our object does not modify the simulation state or an object's attributes, but writes the timestep to standard out, our action is a writer. 
# Create a CustomWriter operation that will call the custom action when triggered:
custom_op = hoomd.write.CustomWriter(
    action=custom_action, trigger=hoomd.trigger.Periodic(100)
)

# To use a custom operation we must add it to a hoomd.Operations object. Thus, the steps to use a custom action in a simulation are:

# 1) Instantiate the custom action object.
# 2) Wrap the custom action in the appropriate custom operation class.
# 3) Add the custom operation object to the appropriate container in a hoomd.Operations object.

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu)
simulation.create_state_from_snapshot(snap)
simulation.operations.writers.append(custom_op)
simulation.run(1000)

# Let's create a new action & simulation
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu)

snap = hoomd.Snapshot()
snap.particles.N = 1
snap.particles.position[:] = [0, 0, 0]
snap.particles.types = ["A"]
snap.particles.typeid[:] = [0]
snap.configuration.box = [10, 10, 10, 0, 0, 0]

simulation.create_state_from_snapshot(snap)

# By the time that a custom action will have its act method called it will have an attribute _state accessible to it which is the simulation state for the simulation it is associated with. 
# The behavior of this is controlled in the hoomd.custom.Action.attach method. 
# The method takes in a simulation object and performs any necessary set-up for the action call act. 
# By default, the method stores the simulation state in the _state attribute.

# We will create two custom actions class to show this. 
# In one, we will not modify the attach method, and in the other we will make attach method also print out some information.
class PrintTimestepNew(hoomd.custom.Action):
    def act(self, timestep):
        print(timestep)


class NotifyAttachWithPrint(hoomd.custom.Action):
    def attach(self, simulation):
        print(f"Has '_state' attribute {hasattr(self, '_state')}.")
        super().attach(simulation)
        print(f"Has '_state' attribute {hasattr(self, '_state')}.")

    def act(self, timestep):
        print(timestep)

# Like in the previous section these are both writers. 
# We will go ahead and wrap them and see what happens when we try to run the simulation.
print_timestep = PrintTimestepNew()
print_timestep_operation = hoomd.write.CustomWriter(
    action=print_timestep, trigger=hoomd.trigger.Periodic(10)
)
simulation.operations.writers.append(print_timestep_operation)
simulation.run(0)

simulation.operations -= print_timestep_operation
print_timestep_with_notify = NotifyAttachWithPrint()
simulation.operations.writers.append(
    hoomd.write.CustomWriter(
        action=print_timestep_with_notify, trigger=hoomd.trigger.Periodic(10)
    )
)
simulation.run(0)

# Custom actions can hook into HOOMD-blue's logging subsystem by using the hoomd.logging.log decorator to document which methods/properties of a custom action are loggable. 
# See the documentation on hoomd.logging.log and hoomd.logging.TypeFlags for complete documenation of the decorator and loggable types.
# In general, log as a decorator takes optional arguments that control whether to make a method a property, what type the loggable quantity is, and whether the quantity should be logged by default.
class ActionWithLoggables(hoomd.custom.Action):
    @hoomd.logging.log
    def scalar_property_loggable(self):
        return 42

    @hoomd.logging.log(category="string")
    def string_loggable(self):
        return "I am a string loggable."

    def act(self, timestep):
        pass

action = ActionWithLoggables()
print(action.scalar_property_loggable)
print(action.string_loggable)


# Another feature of the custom action API is that when an object is wrapped by a custom operation object (which is necessary to add a custom action to a simulation), the action's attributes are available through the operation object as if the operation were the action. 
# For example, we will wrap action from the previous code block in a CustomWriter and access its attributes that way.
custom_op = hoomd.write.CustomWriter(action=action, trigger=100)
print(custom_op.scalar_property_loggable)
print(custom_op.string_loggable)