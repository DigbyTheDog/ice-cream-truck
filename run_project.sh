#!/bin/bash

source venv/bin/activate

rm captured_image.png
rm isolated_drawing.png
rm gumball_locations.json

echo "Taking photo..."
python3 src/take_photo.py

if ! [ -e "captured_image.png" ];
	then echo "Failed to take photo. Exiting.";
	exit 1
fi

echo "Starting image processing..."
python3 src/image_processing.py "captured_image.png"

if ! [ -e "isolated_drawing.png" ];
	then echo "Failed to isolate drawing from photo. Exiting.";
	exit 1
fi

echo "Starting rendering in Blender..."
blender --background src/Popsicle.blend --python src/render_script.py