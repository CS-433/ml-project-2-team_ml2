#!/usr/bin/python


# Import libraries
# TODO Importer depuis src... @Anthony J'ai pas trouvé comment faire, du coup j'ai déplacé les fichiers .py dans root
from load_data import *

# Load data
input_dir_obs = "Data/training_processed/images"
input_dir_label = "Data/training_processed/groundtruth"
obs = load_data(input_dir_obs, img_size=400)
label = load_data(input_dir_label, img_size=400)

# Label to binary, unique channel
label = label[:, 1] > 0
label = label[:, None, :, :]  # Ajoute la dim C = 1

# Initialize neural network
# U_net

# Train neural network

# Make-save prediction
