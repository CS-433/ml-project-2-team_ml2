"""
    This script is the second step of the project. It is used to preprocess loaded data.
"""
import cv2
import os

def create_path(dirName):
        # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

def resize_image(height, width, imgs):
    imgs = [cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA) for img in imgs]
    return imgs