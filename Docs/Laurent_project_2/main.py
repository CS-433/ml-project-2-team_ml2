"""
    This script is the second step of the project. It is used to preprocess loaded data.
"""

"""
    Libraries that we might want to use. Not sure all are used.
"""
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from data_loader import *
from data_processor import *
from model import UNet
from predict import *
from train import *
import segmentation_models as sm
from config import MODEL_NAME
from mask_to_submission import *
from data_augmentation import *
import matplotlib.cm as cm


seed = 0

def main():
    # Load datasets
    imgs = load_train_data_augmented()
    gt_imgs = load_train_labels_augmented()
    test_imgs = load_test_data()
    
    # Resize images
    imgs = resize_image(256, 256, imgs)
    gt_imgs = resize_image(256, 256, gt_imgs)
    gt_imgs = np.expand_dims(gt_imgs, -1)
    test_imgs = resize_image(256, 256, test_imgs)
    imgs = [tf.image.convert_image_dtype(img, dtype=tf.float32) for img in imgs]
    gt_imgs = [tf.image.convert_image_dtype(img, dtype=tf.float32) for img in gt_imgs]
    test_imgs = [tf.image.convert_image_dtype(img, dtype=tf.float32) for img in test_imgs]
    imgs = np.asarray(imgs)
    gt_imgs = np.asarray(gt_imgs)
    test_imgs = np.asarray(test_imgs)
    
    # Split for training
    train_images, test_images, train_masks, test_masks = train_test_split(imgs, gt_imgs, test_size=0.2, random_state=seed)
    
    # Build model
    print("\n### building model ###\n")
    if MODEL_NAME == "UNet":
        model = UNet()
    elif MODEL_NAME == "efficientnetb0":
        sm.set_framework('tf.keras')
        sm.framework()
        model = sm.Unet(backbone_name='efficientnetb0', encoder_weights='imagenet', encoder_freeze=False)

  
    # Train model
    print("\n### training model ###\n")
    train(model, train_images, train_masks, EPOCHS, LEARNING_RATE, BATCH_SIZE, DEFAULT_OUT_PATH, DEFAULT_FINAL_OUT_PATH) # I don't know the purpose of 2 different models but this is how it is done (maybe we can find another way ?)

    # Predictions
    print(f"\n### getting predictions of {len(test_imgs)} test images ###\n")
    preds_test = get_predictions("./Models/model_final", test_images)
    preds = get_predictions("./Models/model_final", test_imgs)

    #Display predictions
    print("\n### displaying predictions ###")
    display_predictions(preds_test, THRESH_VAL, test_images, NUM_SAMPLES, test_masks)

    #Resize preds_test for submission
    preds = resize_image(608, 608, preds)

    #Create images for submission
    for i in range(1, 51):
        plt.imsave('Predictions/satImage_' + '%.3d' % i + '.png', preds[i - 1].squeeze(), cmap=cm.gray)


    print("\n### end of main ###")

if __name__ == '__main__':
    create_path("./Augmented/images")
    create_path("./Augmented/groundtruth")
    create_path("./Predictions")
    images = load_train_data()
    labels = load_train_labels()

    data = tf.data.Dataset.from_tensor_slices({"image": images, "mask":labels})
    augment_data(data)
    main()
    submission_filename = 'final_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'Predictions/satImage_' + '%.3d' % i + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)