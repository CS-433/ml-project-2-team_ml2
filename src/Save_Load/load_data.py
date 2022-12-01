#!/usr/bin/python
import numpy as np
# Import necessary libraries
import torch
from torch.utils.data import Dataset
from torchvision.transforms import FiveCrop
import torchvision as tv
import os


class Roads(Dataset):
    # mapping between label class names and indices
    def __init__(self, split='train', img_size=400,frac_data='1.0'):

        # prepare data
        self.img_size = img_size
        # get images with correct index according to dataset split

        if split == 'train' or split == 'val':
            input_dir_obs = "Data/training_processed/images"
            input_dir_label = "Data/training_processed/groundtruth"
            obs = load_data(input_dir_obs, img_size, split, frac_data)
            label = load_data(input_dir_label, img_size, split, frac_data)

            # Label to binary, unique channel
            label = label[:, 1] > 0
            label = label[:, None, :, :]  # Ajoute la dim C = 1
            self.data = []
            for i in range(label.shape[0]):
                self.data.append((obs[i,:],label[i,:]))
            #self.data = obs, label

        if split == 'test':
            input_dir_test = "Data/test_set_images"
            up_lefts, up_rights, down_lefts, down_rights = load_test_data(input_dir_test)
            test = torch.cat((up_lefts, up_rights, down_lefts, down_rights))
            n_image = len(test)
            label = torch.zeros(n_image, 1, img_size, img_size, dtype=torch.uint8)  # Bonne initialization ?

            self.data = test, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        img, label = self.data[x]
        return img, label


def load_data(input_dir, img_size=400,split = 'train', frac_data = '1.0'):
    """
    For the observation :
        Convert all the images from the directory into a tensor of size (N, C, H, W)

    Returns the tensor

    Parameters
    ----------
    input_dir : str
        path of directory containing images
    img_size : int
        size of the border of the image (assumed squared)

    Returns
    -------
    tensor : torch.Tensor
        all images of the directory as a tensors of size (N, C, H, W)

    """
    filenames = sorted([name for name in os.listdir(input_dir)])

    if split == 'train':
        n_image = np.floor(0.8*len(filenames))
    elif split == 'val':
        n_image = len(filenames) - np.floor(0.8 * len(filenames))
    else:
        raise TypeError("Split parameter not recognised!")
    n_image = int(n_image*frac_data)

    if split == 'train':
        filenames = filenames[0:n_image]
    elif split == 'val':
        filenames = filenames[-n_image:]
    n_channel = 3
    print(len(filenames))
    print(n_image)
    tensor = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        tensor[i] = tv.io.read_image(os.path.join(input_dir, filename))
    return tensor


def load_test_data(rootdir, img_size=400):
    """
    Returns 4 tensors of images of size (N x C x H x W)

    with H and W = 400
                C = 3
                N = number of test images


    The 4 tensor are for the 4 different parts of the test image (up_left, up_right, down_left, down_right)
    """

    dirs = [os.path.join(rootdir, file) for file in os.listdir(rootdir)]

    n_image = len(dirs)
    n_channel = 3

    up_lefts = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    up_rights = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    down_lefts = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    down_rights = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)

    for i, d in enumerate(dirs):
        image_test = load_data(d, img_size=608)
        transform = FiveCrop(img_size)
        up_left, up_right, down_left, down_right, _ = transform(image_test)
        up_lefts[i] = up_left
        up_rights[i] = up_right
        down_lefts[i] = down_left
        down_rights[i] = down_right

    return up_lefts, up_rights, down_lefts, down_rights
