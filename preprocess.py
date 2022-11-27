#!/usr/bin/python

"""
Preprocess the training data (obs and labels) with data augmentation (flip, rotation, ...) and normalization. Saves
the processed data into directory Data/training_processed/
"""

# Import necessary libraries
from load_data import *
from save_data import *

# Paths
input_dir_obs = "Data/training/images/"
input_dir_label = "Data/training/groundtruth/"
output_dir_obs = "Data/training_processed/images/"
output_dir_label = "Data/training_processed/groundtruth/"

# Load data
obs = load_data(input_dir_obs, is_label=False, img_size=400)
label = load_data(input_dir_label, is_label=True, img_size=400)

# Labels to value 0 or 255 (no value in between)
label = label >= 128
label = (label*255).to(dtype=torch.uint8)

# TODO Stack obs and labels and apply common transformation to the stacked tensor (flip, rotation, normalize)
# TODO Add other transform such as ???

# Flip
flip_obs = tv.transforms.functional.hflip(obs)
obs = torch.cat((obs, flip_obs))

flip_label = tv.transforms.functional.hflip(label)
label = torch.cat((label, flip_label))

# Rotate
rot_obs_90 = tv.transforms.functional.rotate(obs, 90)
rot_obs_180 = tv.transforms.functional.rotate(obs, 180)
rot_obs_270 = tv.transforms.functional.rotate(obs, 270)
obs = torch.cat((obs, rot_obs_90, rot_obs_180, rot_obs_270))

rot_label_90 = tv.transforms.functional.rotate(label, 90)
rot_label_180 = tv.transforms.functional.rotate(label, 180)
rot_label_270 = tv.transforms.functional.rotate(label, 270)
label = torch.cat((label, rot_label_90, rot_label_180, rot_label_270))

# Noise TODO Add noise

# Normalize
obs = obs/255
label = label/255

# Save data as images
save_data(obs, output_dir_obs)
save_data(label, output_dir_label)
