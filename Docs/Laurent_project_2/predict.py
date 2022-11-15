"""
    This script is the final step. It is used to test the model, i.e. make predictions.
"""
from keras.models import load_model
from train import *
import numpy as np
import random
import matplotlib.pyplot as plt

def get_predictions(model_path, test_images):
    model = load_model(model_path, custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef})
    predictions = model.predict(test_images, verbose=1)
    return predictions

def display_predictions(predictions, thresh_val, test_images, num_samples, test_masks):
    predicton_threshold = (predictions > thresh_val).astype(np.uint8)
    print(predictions.shape)

    f = plt.figure(figsize=(30, 50))
    for i in range(1, num_samples, 4):
        ix = random.randint(0, 20)

        f.add_subplot(num_samples, 4, i)
        plt.imshow(test_images[ix])
        plt.title("Image")
        plt.axis('off')

        f.add_subplot(num_samples, 4, i + 1)
        plt.imshow(np.squeeze(test_masks[ix]))
        plt.title("Groud Truth")
        plt.axis('off')

        f.add_subplot(num_samples, 4, i + 2)
        plt.imshow(np.squeeze(predictions[ix]))
        plt.title("Prediction")
        plt.axis('off')

        f.add_subplot(num_samples, 4, i + 3)
        plt.imshow(np.squeeze(predicton_threshold[ix]))
        plt.title("thresholded at {}".format(thresh_val))
        plt.axis('off')

    plt.savefig("./Results/results.png")