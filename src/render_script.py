import bpy, json, random

file_path = 'isolated_drawing.png'
popsicle_width = 0.3
move_up = 0.2


def place_gumballs_from_json(json_path):
    # Load the green sticker positions
    with open(json_path, "r") as f:
        positions = json.load(f)

    gumball = bpy.data.objects['Gumball']

    # Get the plane's size and position
    plane_width = plane.dimensions.x
    plane_height = plane.dimensions.y
    plane_origin = plane.location

    print(f"Plane size: {plane_width} x {plane_height}")
    print(f"Plane origin: {plane_origin}")

    # Iterate over positions and place gumballs
    for i, pos in enumerate(positions):
        image_x = pos["x"]
        image_y = pos["y"]

        # Convert image coordinates to plane-relative coordinates
        plane_x = plane_origin.x + (image_x / 300) * plane_width  # Scale image space to Blender space
        plane_y = plane_origin.y + (1 - (image_y / 300)) * plane_height  # Flip Y-axis

        # Duplicate the gumball
        for i in range(0, random.randint(0, 50)): # This is a really lazy way to get a random color for the gumball
            gumball.copy()
        gumball_copy = gumball.copy()
        gumball_copy.location = (plane_origin.z + 0.11, plane_x - 0.5, plane_y - 0.27)  # Need these stupid offsets for now
        bpy.context.collection.objects.link(gumball_copy)
        gumball_copy.name = f"Gumball_{i + 1}"

    print(f"Placed {len(positions)} gumballs in the scene.")


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

# Place the gum balls
place_gumballs_from_json("gumball_locations.json")

# Set the render output path
bpy.context.scene.render.filepath = "rendered_image.png"

# Render the scene
bpy.ops.render.render(write_still=True)