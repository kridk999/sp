import copy
from email.mime import image
from math import inf
import os
import re
import numpy as np
import SimpleITK as sitk
import json
from scipy.spatial import KDTree
from skimage.morphology import binary_dilation
from scipy.ndimage import center_of_mass
from platipy.imaging.label.utils import get_com
from platipy.imaging.utils.crop import crop_to_roi, label_to_roi
from platipy.imaging.utils.geometry import vector_angle
from platipy.imaging.utils.valve import generate_valve_using_cylinder
from pathlib import Path
import imageio

import utils

def create_gif(filenames, save_folder, save_name, delete_files=False):
    save_name = save_name if save_name.endswith(".gif") else save_name + ".gif"
    with imageio.get_writer(os.path.join(save_folder, save_name), fps=8, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            if delete_files:
                os.remove(filename)
    print(f"GIF saved as {save_name}")
    
if __name__ == "__main__":
    filename_path = "assets/data/0010/images/gif_LCX"
    filename_path = "assets/data/0010/images/bullseye"
    file_order_path = "assets/data/0010/raw/DTU-CFA-Pilot-1-CFA-1.txt"
    
    # Read the txt file to get the order
    with open(file_order_path, 'r') as f:
        file_order = f.readlines()
    
    # Extract last two digits from each line and create ordered list
    lines = []
    for line in file_order:
        line = line.strip()
        if line:  # Skip empty lines
            # Get last two digits
            last_two_digits = line[-2:]
            # Check if corresponding png file exists
            png_file = os.path.join(filename_path, f"bullseye_0010_00{last_two_digits}.png")
            if os.path.exists(png_file):
                lines.append(last_two_digits)
            
    create_gif(
        filenames=[os.path.join(filename_path, f"bullseye_0010_00{line}.png") for line in lines],
        save_folder="assets/data/0010/images",
        save_name="Bullseye.gif",
        delete_files=False
    )


