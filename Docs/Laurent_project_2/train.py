"""
    This script is the main part of the project. Here is where the model is trained.
"""
from config import *
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
"""
    Some metrics for the train.
"""

from keras import backend as K

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


"""
    Training of the model.
"""


def train(model, train_images, train_masks, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, model_path = DEFAULT_OUT_PATH, output_path=DEFAULT_FINAL_OUT_PATH):

    checkpointer = ModelCheckpoint(model_path,
                 monitor="val_loss",
                      mode="min",
                      save_best_only = True,
                          verbose=1)

    earlystopper = EarlyStopping(monitor = 'val_loss', 
                              min_delta = 0, 
                              patience = 5,
                              verbose = 1,
                              restore_best_weights = True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4)
    
    opt = tf.keras.optimizers.Adam(learning_rate)

    model.compile(optimizer=opt, loss=soft_dice_loss, metrics=[iou_coef])
    tensorboard = TensorBoard(log_dir="./logs")
    model.fit(train_images,
                train_masks/255,
                validation_split = 0.1,
                epochs=epochs,
                batch_size = batch_size,
                callbacks = [earlystopper, lr_reducer, checkpointer, tensorboard])
    
    model.save(output_path)
    