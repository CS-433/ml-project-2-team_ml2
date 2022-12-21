#!/usr/bin/python

# Import necessary libraries
import torchvision as tv


def save_data(tensor, output_dir, target=True):
    """
    Saves the tensor as png files in the output_dir
    """
    tensor = tensor.float()
    for i, t in enumerate(tensor):
        if target:
            filename = output_dir + ("0" if i < 9 else "") + str(i + 1) + "_target.png"
        else:
            filename = output_dir + ("0" if i<9 else "") +str(i+1) + "_output.png"
        tv.utils.save_image(t, filename)
