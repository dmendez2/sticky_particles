from ovito.io import import_file
from ovito.vis import Viewport
from ovito.modifiers import ColorCodingModifier

# Load the GSD file
pipeline = import_file("trajectories/single_patch/lm_25_0/mxene_0_25/delta_5.gsd")

# Color particles by type
modifier = ColorCodingModifier(property='Particle Type', start_value=0, end_value=1)
pipeline.modifiers.append(modifier)

# Add the pipeline to the scene so we can render it
pipeline.add_to_scene()

# Create a viewport for rendering
viewport = Viewport()
viewport.type = Viewport.Type.Perspective  # Optional: can also use Top, Bottom, etc.
viewport.render_image(size=(800, 600), filename="frame.png", background=(1, 1, 1))
