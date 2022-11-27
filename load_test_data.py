#!/usr/bin/python
import os
from load_data import *
from torchvision.transforms.functional import five_crop

def load_test_data():

    rootdir = "./Data/test_set_images"
    dirs = [os.path.join(rootdir, file) for file in os.listdir(rootdir)]
    for dir in dirs:
        image_test = load_data(dir, is_label=False, img_size=608)
        image_test_corners = five_crop(image_test, 400)[:-1]

    raise NotImplementedError