import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.initializers import RandomNormal

def build_discriminator(img_shape = [256, 256, 1], lr = 0.0002, b = [0.5, 0.0], drop =0.1):

    init = RandomNormal(stddev = 0.02)
    #init = he_normal()
    input_img = Input(shape = img_shape)
    input_tar = Input(shape = img_shape)
    input = Concatenate()([input_img, input_tar])

    d = Conv2D(64, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(input)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(128, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(256, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(512, (4,4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(512, (4,4), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d = Conv2D(1, (4,4), padding = 'same', kernel_initializer = init)(d)
    d = Dropout(drop)(d, training = True)
    out = Activation('sigmoid')(d)

    model = Model([input_img, input_tar], out)
    #opt = Adam(lr = lr, beta_1 = b[0])
    #opt = RMSprop(lr = lr, momentum = 0.9)
    opt = RMSprop(learning_rate = lr, momentum = 0.9)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, loss_weights = [0.5])

    return model

if __name__ == '__main__':

    model = build_discriminator()
    model.summary()
