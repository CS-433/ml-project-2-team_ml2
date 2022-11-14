###
# USING TENSORFLOW
    # This script outputs as many images as in the input. E.g. for 5 original images, we get 5 transformed images.
    # So we need to run it many times (as many as needed) to get more transformed images.
###
# necessary imports
from config import TRAIN_IMG_DIR_AUG
from config import TRAIN_GT_DIR_AUG
import tensorflow as tf
import numpy as np
from functools import partial
import albumentations as A
from data_loader import load_train_data, load_train_labels
from config import NUMBER_AUGMENTED_IMG
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomContrast(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])


def aug_fn(image, mask):
    _data = {"image":image, "mask":mask}
    aug_data = transforms(**_data)
    aug_img = aug_data["image"]
    aug_mask = aug_data["mask"]
    return aug_img, aug_mask

def process_data(_tuple):
    image = _tuple['image']
    mask = _tuple['mask']
    aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, mask], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask


def augment_data(data, N=int(NUMBER_AUGMENTED_IMG/100)):
    imgs_path = TRAIN_IMG_DIR_AUG
    masks_path = TRAIN_GT_DIR_AUG
    
    for i in range(N):
        print(f"Loop {i+1}/{N}")
        # transforming
        ds_alb = data.map(partial(process_data),
                         num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)        
        # saving
        idx = 0
        for e in ds_alb:
            image = e[0].numpy()
            mask = e[1].numpy()
            img_file = imgs_path+str(i)+'_'+str(idx)+'.png'
            Image.fromarray((image*255).astype(np.uint8)).save(img_file)
            #plt.imsave(fname=img_file, arr=image)
            mask_file = masks_path+str(i)+'_'+str(idx)+'.png'
            #plt.imsave(fname=mask_file, arr=mask, cmap='gray')
            Image.fromarray((mask*255).astype(np.uint8)).save(mask_file)
            idx = idx+1
    
if __name__ == "__main__":
    # loading
    images = load_train_data()
    labels = load_train_labels()

    data = tf.data.Dataset.from_tensor_slices({"image": images, "mask":labels})
    augment_data(data)

