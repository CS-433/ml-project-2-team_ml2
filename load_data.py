#!/usr/bin/python

# Import necessary libraries
import torch
import torchvision as tv
import os




def load_data(input_dir, is_label=False, img_size=400):
    """
    For the observation :
        Convert all the images from the directory into a tensor of size (N, C, H, W)

    Returns the tensor

    Parameters
    ----------
    input_dir : str
        path of directory containing images
    is_label : bool
    img_size : int
        size of the border of the image (assumed squared)

    Returns
    -------
    tensor : torch.Tensor
        all images of the directory as a tensors of size (N, C, H, W)

    """
    filenames = sorted([name for name in os.listdir(input_dir)])
    n_image = len(filenames)
    print(filenames)

    if is_label:
        n_channel = 1
    else:
        n_channel = 3

    tensor = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        if is_label:
            print(tensor.shape)
            test = tv.io.read_image(os.path.join(input_dir, filename))
            print(test.shape)
        tensor[i] = tv.io.read_image(os.path.join(input_dir, filename))
        return tensor
