#!/usr/bin/python


# Import libraries
# TODO Importer depuis src... @Anthony J'ai pas trouvé comment faire, du coup j'ai déplacé les fichiers .py dans root
from load_data import *

# Load data
input_dir_obs = "Data/training_processed/images"
input_dir_label = "Data/training_processed/groundtruth"
obs = load_data(input_dir_obs, is_label=False, img_size=400)
label = load_data(input_dir_label, is_label=True, img_size=400)

# Initialize neural network
# U_net

# Train neural network

# Make-save prediction
