#!/usr/bin/python

# Import necessary libraries
import torchvision as tv


def save_data(tensor, output_dir):
    """
    Saves the tensor as png files in the output_dir
    """
    tensor = tensor.float()
    for i, t in enumerate(tensor):
        filename = output_dir + str(i+1) + ".png"
        tv.utils.save_image(t, filename)
