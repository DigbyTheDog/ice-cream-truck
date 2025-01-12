#!/bin/bash

echo "Starting image processing..."
python3 src/image_processing.py input_image.png output_image.png

echo "Starting rendering in Blender..."
blender --background src/Popsicle.blend --python src/render_script.py