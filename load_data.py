#!/usr/bin/python

# Import necessary libraries
import torch
from torch.utils.data import Dataset
import torchvision as tv
import os


class Roads(Dataset):
    # mapping between label class names and indices
    LABEL_CLASSES = {
        'rocks': 0,
        'scree': 1,
        'sparse_rocks': 2,
        'water': 3,
        'glacier_and_permanent_snow': 4,
        'forest': 5,
        'sparse_forest': 6,
        'grasslands_and_others': 7,

    }

    def __init__(self, split='train', img_size=400):

        # prepare data
        self.data = []  # list of tuples of (image path, label class)
        self.img_size = img_size
        # get images with correct index according to dataset split
        data_names = ""
        gt_names = ""
        data_fold = ""
        gt_fold = ""
        if split == 'train':
            data_fold = "Data/training_processed/images"
            gt_fold = "Data/training_processed/groundtruth"
            data_names = sorted([name for name in os.listdir(data_fold)])
            #gt_names = sorted([name for name in os.listdir(gt_fold)])
        if split == 'test':
            data_fold = "Data/test_set_images"
            gt_fold = "Data/test_set_images/groundtruth"
            data_names = sorted([name for name in os.listdir(data_fold)])
            #gt_names = sorted([name for name in os.listdir(gt_fold)])

        #load data
        #n_images = len(data_names)
        #n_channel = 3
        #self.data = torch.zeros(n_images, n_channel, img_size, img_size, dtype=torch.uint8)
        for i, filename in enumerate(data_names):
            #self.data[i] = tv.io.read_image(os.path.join(data_fold, filename))
            img = tv.io.read_image(os.path.join(data_fold, filename))
            label = tv.io.read_image(os.path.join(gt_fold, filename))
            self.data.append((img, label))

        #load labels
        """n_labels = len(data_names)
        n_channel = 3
        self.label = torch.zeros(n_labels, n_channel, img_size, img_size, dtype=torch.uint8)
        for i, filename in enumerate(gt_names):
            self.data[i] = tv.io.read_image(os.path.join(gt_fold, filename))"""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        img, label = self.data[x]
        return img, label

        #img = Image.open(img_name)
        #img_label = Image.open(img_label_name)

        #if self.transforms is not None:
        #    img = self.transforms(img)
        #    img_label = self.transforms_label(img_label)
        #return img, img_label



def load_data(input_dir, img_size=400):
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
    n_image = len(filenames)
    n_channel = 3

    tensor = torch.zeros(n_image, n_channel, img_size, img_size, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        tensor[i] = tv.io.read_image(os.path.join(input_dir, filename))
    return tensor
