#!/bin/bash

echo "Updating package lists..."
sudo apt-get update

echo "Installing Python and pip..."
sudo apt-get install -y python3 python3-pip

echo "Installing OpenCV and other dependencies..."
pip3 install -r requirements.txt

echo "Installing Blender..."
sudo apt-get install -y blender

echo "Setup complete!"