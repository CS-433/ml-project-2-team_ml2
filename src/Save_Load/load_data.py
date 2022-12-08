#!/usr/bin/python
import numpy as np
# Import necessary libraries
import torch
from torch.utils.data import Dataset
from torchvision.transforms import FiveCrop
import torchvision as tv
import os
import matplotlib.image as mpimg


class Roads(Dataset):
    # mapping between label class names and indices
    def __init__(self, split='train', img_size=400, frac_data=1.0):

        # prepare data
        self.img_size = img_size
        # get images with correct index according to dataset split
        if split == 'train' or split == 'val':
            input_dir_obs = "Data/training_processed/images"
            input_dir_label = "Data/training_processed/groundtruth"
            # Load data from files:
            obs = load_data(input_dir_obs, img_size, split, frac_data)
            label = load_data(input_dir_label, img_size, split, frac_data)

            # Label to binary, unique channel
            label = label[:, 1] > 0
            label = label[:, None, :, :]  # Ajoute la dim C = 1
            self.data = []
            for i in range(label.shape[0]):
                self.data.append((obs[i, :], label[i, :]))

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


def load_data(input_dir, img_size=400, split='train', frac_data=1.0):
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

    # ===== DEFINITION OF NUMBER OF IMAGES TO IMPORT =====
    if split == 'train':
        n_image = np.floor(0.8 * len(filenames))
    elif split == 'val':
        n_image = len(filenames) - np.floor(0.8 * len(filenames))
    elif split == 'test':
        n_image = len(filenames)
    else:
        raise TypeError("Split parameter not recognised!")
    n_image = int(n_image * frac_data)

    # ===== SELECTION OF THE FILE NAMES TO IMPORT =====
    if split == 'train':
        filenames = filenames[0:n_image]
    elif split == 'val':
        filenames = filenames[-n_image:]

    # ===== IMPORTATION OF THE IMAGES =====
    n_channel = 3
    tensor = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        tensor[i] = tv.io.read_image(os.path.join(input_dir, filename))
    return tensor


def load_test_data(rootdir, img_size=400):
    """
    Returns a tensor of images of size ( 4N x C x H x W) from the rootdir of the test_set_images

    with H and W = 400
                C = 3
                N = number of test images


    The tensor containes 4 different parts of each test image (up_left, up_right, down_left, down_right)
    """

    dirs = [os.path.join(rootdir, file) for file in os.listdir(rootdir)]

    n_image = len(dirs)
    n_channel = 3

    test_tensor = torch.zeros(4 * n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    index = 0

    for i, d in enumerate(dirs):
        image_test = load_data(d, img_size=608, split="test")
        transform = FiveCrop(img_size)
        up_left, up_right, down_left, down_right, _ = transform(image_test)

        test_tensor[index] = up_left
        index += 1
        test_tensor[index] = up_right
        index += 1
        test_tensor[index] = down_left
        index += 1
        test_tensor[index] = down_right
        index += 1

    return test_tensor


def fuse_four_corners_labels(four_corners_labels, in_size=400, out_size=608, n_channel=1):
    assert out_size <= 2 * in_size

    n_corners = four_corners_labels.size(dim=0)
    assert n_corners % 4 == 0
    n_images = n_corners // 4
    whole_labels = torch.zeros(n_images, n_channel, out_size, out_size, dtype=torch.bool)
    for i in range(n_images):
        index = i * 4
        up_left = four_corners_labels[index]
        up_right = four_corners_labels[index + 1]
        low_left = four_corners_labels[index + 2]
        low_right = four_corners_labels[index + 3]

        whole_labels[i, :, 0:in_size, 0:in_size] = up_left
        whole_labels[i, :, 0:in_size, out_size - in_size:out_size] = up_right
        whole_labels[i, :, out_size - in_size:out_size, 0:in_size] = low_left
        whole_labels[i, :, out_size - in_size:out_size, out_size - in_size:out_size] = low_right

    return whole_labels


def load_test_data_2(test_dir):  # CHECKED
    test_names = os.listdir(test_dir)

    prefixes = ('.')
    for dir_ in test_names[:]:
        if dir_.startswith(prefixes):
            test_names.remove(dir_)
    num_test = len(test_names)

    # get data permutation
    order = [int(test_names[i].split("_")[1]) for i in range(num_test)]
    p = np.argsort(order)
    n_channel = 3
    img_size = 608
    tensor = torch.zeros(50, n_channel, img_size, img_size, dtype=torch.uint8)
    for i, filename in enumerate(test_names):
        tensor[i] = tv.io.read_image(os.path.join(test_dir, test_names[i], test_names[i]) + ".png")

    return tensor


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data
