import tensorflow as tf
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
from keras.models import Sequential
            
def block(input_shape, filter_size, init_dropout, name):
    model = Sequential()
    model.add(Conv2D(
        filters=input_shape,
        kernel_size=filter_size,
        kernel_initializer='he_normal',
        padding='same',
        activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(init_dropout))
    model.add(Conv2D(
        filters=input_shape,
        kernel_size=filter_size,
        kernel_initializer='he_normal',
        padding='same',
        activation='elu'))
    model.add(BatchNormalization())
    return model

class UNet(tf.keras.Model):

    def __init__(self, input_shape=16, init_dropout=0.1):
        super(UNet, self).__init__()
        self.model_name = "unet"
        filters_size_Conv2D = (3,3)
        filters_size_Conv2DTranspose = (2,2)
        self.encoder1 = block(input_shape, filters_size_Conv2D, init_dropout, name="encoder1")
        self.pool1 = MaxPooling2D((2, 2))
        
        self.encoder2 = block(input_shape * 2, filters_size_Conv2D, init_dropout, name="encoder2")
        self.pool2 = MaxPooling2D((2, 2))
        
        self.encoder3 = block(input_shape * 4, filters_size_Conv2D, init_dropout * 2, name="encoder3")
        self.pool3 = MaxPooling2D((2, 2))
        
        self.encoder4 = block(input_shape * 8, filters_size_Conv2D, init_dropout * 2, name="encoder4")
        self.pool4 = MaxPooling2D((2, 2))

        self.bottom = block(input_shape * 16, filters_size_Conv2D, init_dropout * 3, name="bottom")

        self.upconv4 = Conv2DTranspose(
            input_shape * 8, filters_size_Conv2DTranspose, strides=(2,2), padding = "same"
        )
        self.decoder4 = block(input_shape * 8, filters_size_Conv2D, init_dropout * 2, name="decoder4")
        
        self.upconv3 = Conv2DTranspose(
            input_shape * 4, filters_size_Conv2DTranspose, strides=(2,2), padding = "same"
        )
        self.decoder3 = block(input_shape * 4, filters_size_Conv2D, init_dropout * 2, name="decoder3")
        
        self.upconv2 = Conv2DTranspose(
            input_shape * 2, filters_size_Conv2DTranspose, strides=(2,2), padding = "same"
        )
        self.decoder2 = block(input_shape * 2, filters_size_Conv2D, init_dropout, name="decoder2")
        
        self.upconv1 = Conv2DTranspose(
            input_shape, filters_size_Conv2DTranspose, strides=(2,2), padding = "same"
        )
        self.decoder1 = block(input_shape, filters_size_Conv2D, init_dropout, name="decoder1")

        self.finalconv = Conv2D(
            1, (1, 1), activation='sigmoid'
        )

    def call(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottom = self.bottom(self.pool4(enc4))

        dec4 = self.upconv4(bottom)
        dec4 = concatenate([dec4, enc4])
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = concatenate([dec3, enc3])
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = concatenate([dec2, enc2])
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = concatenate([dec1, enc1])
        dec1 = self.decoder1(dec1)
        
        return self.finalconv(dec1)