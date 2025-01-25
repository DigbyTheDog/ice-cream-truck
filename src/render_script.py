import bpy

file_path = 'isolated_drawing.png'
popsicle_width = 0.3
move_up = 0.2

bpy.ops.wm.open_mainfile(filepath="src/Popsicle.blend")

bpy.ops.image.import_as_mesh_planes(shader='SHADELESS', files=[{'name':file_path}])

plane = bpy.context.object
if plane is None:
    raise RuntimeError("No plane was imported. Check your image path.")

material_name = "Popsicle"
if material_name in bpy.data.materials:
    material = bpy.data.materials[material_name]
else:
	raise RuntimeError("Popsicle Material not found. Check materials.")#

# Assign the material to the popsicle
plane.data.materials.clear()
plane.data.materials.append(material)

# Change the image on the material
node = material.node_tree.nodes["Image Texture"]
img = bpy.data.images[file_path]
node.image = img

# Extrude that bad boy
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.extrude_region()
bpy.ops.transform.translate(value=(popsicle_width, 0, 0))

# Move it up a bit
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.transform.translate(value=(0, 0, move_up))

# Set the render output path
bpy.context.scene.render.filepath = "rendered_image.png"

# Render the scene
bpy.ops.render.render(write_still=True)