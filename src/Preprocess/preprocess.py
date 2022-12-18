#!/usr/bin/python

"""
Preprocess the training data (obs and labels) with data augmentation (flip, rotation, ...) and normalization. Saves
the processed data into directory Data/training_processed/
"""

# Import necessary libraries
from src.Save_Load.load_data import *
from src.Save_Load.save_data import *
import torchvision as tv

# Paths
input_dir_obs = "../../Data/training/images/"
input_dir_label = "../../Data/training/groundtruth/"
output_dir_obs = "../../Data/training_processed/images/"
output_dir_label = "../../Data/training_processed/groundtruth/"

# Load data
obs = load_data(input_dir_obs, img_size=400, split="test")
label = load_data(input_dir_label, img_size=400, split="test")

# Labels to value 0 or 255 (no value in between)
label = label >= 128
label = (label * 255).to(dtype=torch.uint8)

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
rot_obs_45 = tv.transforms.functional.rotate(obs, 45, fill=0)
obs = torch.cat((obs, rot_obs_45))

rot_label_90 = tv.transforms.functional.rotate(label, 90)
rot_label_180 = tv.transforms.functional.rotate(label, 180)
rot_label_270 = tv.transforms.functional.rotate(label, 270)
label = torch.cat((label, rot_label_90, rot_label_180, rot_label_270))
rot_label_45 = tv.transforms.functional.rotate(label, 45, fill=0)
label = torch.cat((label, rot_label_45))

# Noise NONE...

# Change type
obs = obs.to(dtype=torch.float)

image_standardization = False
all_standardization = True

# Standardization among each image (with each pixels)
if image_standardization:
    mean_obs = obs.mean(dim=(2, 3), dtype=torch.float)
    sd_obs = obs.std(dim=(2, 3), unbiased=True)

    for i in range(1600):
        for j in range(3):
            mean = mean_obs[i, j]
            std = sd_obs[i, j]
            obs[i, j] = (obs[i, j] - mean) / std

    # mean_obs = obs.mean(dim=(2, 3), dtype=torch.float) ALL 0 -->YES
    # sd_obs = obs.std(dim=(2, 3), unbiased=True)  ALL 1 --> YES

# Standardization through all images
if all_standardization:
    mean_obs = obs.mean(dim=(0), dtype=torch.float)
    sd_obs = obs.std(dim=(0), unbiased=True)

    for i in range(400):
        for j in range(400):
            for k in range(3):
                mean = mean_obs[k, i, j]
                std = sd_obs[k, i, j]
                obs[:, k, i, j] = (obs[:, k, i, j] - mean) / std

    mean_obs = obs.mean(dim=(0), dtype=torch.float)
    sd_obs = obs.std(dim=(0), unbiased=True)

# Save data as images
save_data(obs, output_dir_obs)
save_data(label, output_dir_label)
