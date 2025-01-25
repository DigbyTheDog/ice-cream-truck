#!/bin/bash

INPUT_IMAGE=$1

echo "Starting image processing..."
source venv/bin/activate
python3 src/image_processing.py ${INPUT_IMAGE}

echo "Starting rendering in Blender..."
blender --background src/Popsicle.blend --python src/render_script.py