#!/bin/bash

echo "Updating package lists..."
sudo apt-get update

echo "Installing Python and venv..."
sudo apt-get install -y python3 python3-venv

echo "Creating a virtual environment..."
python3 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing dependencies in the virtual environment..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing Blender..."
sudo snap install blender --channel=4.3.2/stable

echo "Setup complete!"