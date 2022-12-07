"""
    This script is the first step of the project. It loads the input images.
"""
import os
import matplotlib.image as mpimg
import numpy as np

from config import TRAIN_IMG_DIR_AUG
from config import TRAIN_GT_DIR_AUG
from config import TRAIN_IMG_DIR
from config import TRAIN_GT_DIR
from config import TEST_IMG_DIR
from config import NB_IMAGES_TO_LOAD
from skimage import io

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def load_train_data(nb_imgs=NB_IMAGES_TO_LOAD): #CHECKED
    image_dir = TRAIN_IMG_DIR
    files = os.listdir(image_dir)
    imgs = [load_image(image_dir + files[i]) for i in range(100)]
    return np.asarray(imgs)


def load_train_labels(nb_imgs=NB_IMAGES_TO_LOAD): #CHECKED
    gt_dir = TRAIN_GT_DIR
    files = os.listdir(gt_dir)
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(100)]
    return np.asarray(gt_imgs)

def load_image_augmented(infilename, as_gray):
    data = io.imread(infilename, as_gray=as_gray)
    return data

def load_train_data_augmented(): #CHECKED
    image_dir = TRAIN_IMG_DIR_AUG
    files = os.listdir(image_dir)
    imgs = [load_image_augmented(image_dir + files[i], False) for i in range(NB_IMAGES_TO_LOAD)]
    return np.asarray(imgs)


def load_train_labels_augmented(): #CHECKED
    gt_dir = TRAIN_GT_DIR_AUG
    files = os.listdir(gt_dir)
    gt_imgs = [load_image_augmented(gt_dir + files[i], True) for i in range(NB_IMAGES_TO_LOAD)]
    return np.asarray(gt_imgs)
    
    
def load_test_data(test_dir): #CHECKED
    test_dir = TEST_IMG_DIR
    test_names = os.listdir(test_dir)
    
    prefixes = ('.')
    for dir_ in test_names[:]:
        if dir_.startswith(prefixes):
            test_names.remove(dir_)
    num_test = len(test_names)
    
    #get data permutation
    order = [int(test_names[i].split("_")[1]) for i in range(num_test)]
    p = np.argsort(order)
    imgs_test = [load_image(os.path.join(test_dir, test_names[i], test_names[i]) + ".png") for i in range(num_test)]
    
    #order data
    imgs_test = [imgs_test[i] for i in p]
    return imgs_test