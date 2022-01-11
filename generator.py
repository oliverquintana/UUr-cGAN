from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.initializers import RandomNormal

def residual_block(r, f):

    init = RandomNormal(stddev = 0.02)
    r_short = r

    r = Conv2D(f, (3,3), padding = 'same', kernel_initializer = init)(r)
    r = InstanceNormalization()(r)
    r = LeakyReLU(alpha = 0.2)(r)
    r = Conv2D(f, (3,3), padding = 'same', kernel_initializer = init)(r)
    r = InstanceNormalization()(r)

    r = Add()([r_short, r])
    r = LeakyReLU(alpha = 0.2)(r)

    return r

def build_generator(img_shape = [256, 256, 1], drop = 0.1):

    input_img = Input(shape = img_shape)
    init = RandomNormal(stddev = 0.02)
    #init = he_normal()

    # Encoder
    g1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(input_img)
    g1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g1)
    g1 = InstanceNormalization()(g1)
    g1 = LeakyReLU(alpha = 0.2)(g1)

    g2 = Conv2D(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g1)
    g2 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g2)
    g2 = InstanceNormalization()(g2)
    g2 = LeakyReLU(alpha = 0.2)(g2)

    g3 = Conv2D(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g2)
    g3 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g3)
    g3 = InstanceNormalization()(g3)
    g3 = LeakyReLU(alpha = 0.2)(g3)

    g4 = Conv2D(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g3)
    g4 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = init)(g4)
    g4 = InstanceNormalization()(g4)
    g4 = LeakyReLU(alpha = 0.2)(g4)

    # Bottleneck
    g = Conv2D(1024, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g4)
    g = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = init)(g)
    g = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization()(g)
    g = Dropout(drop)(g, training = True)
    g = LeakyReLU(alpha = 0.2)(g)

    # Decoder
    g5 = Conv2DTranspose(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g5 = Add()([g4, g5])
    g5 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g5)
    g5 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g5)
    g5 = InstanceNormalization()(g5)
    g5 = LeakyReLU(alpha = 0.2)(g5)

    g6 = Conv2DTranspose(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g5)
    g6 = Add()([g3, g6])
    g6 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g6)
    g6 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g6)
    g6 = InstanceNormalization()(g6)
    g6 = LeakyReLU(alpha = 0.2)(g6)

    g7 = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g6)
    g7 = Add()([g2, g7])
    g7 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g7)
    g7 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g7)
    g7 = InstanceNormalization()(g7)
    g7 = LeakyReLU(alpha = 0.2)(g7)

    g8 = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g7)
    g8 = Add()([g1, g8])
    g8 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g8)
    g8 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g8)
    g8 = InstanceNormalization()(g8)
    g8 = LeakyReLU(alpha = 0.2)(g8)

    g1 = residual_block(g8, 64)
    g1 = Add()([g8, g1])
    g1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g1)
    g1 = InstanceNormalization()(g1)
    g1 = LeakyReLU(alpha = 0.2)(g1)

    g2 = Conv2D(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g1)
    g2 = Add()([g7, g2])
    g2 = residual_block(g2, 64)
    g2 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g2)
    g2 = InstanceNormalization()(g2)
    g2 = LeakyReLU(alpha = 0.2)(g2)

    g3 = Conv2D(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g2)
    g3 = Add()([g6, g3])
    g3 = residual_block(g3, 128)
    g3 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g3)
    g3 = InstanceNormalization()(g3)
    g3 = LeakyReLU(alpha = 0.2)(g3)

    g4 = Conv2D(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g3)
    g4 = Add()([g5, g4])
    g4 = residual_block(g4, 256)
    g4 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = init)(g4)
    g4 = InstanceNormalization()(g4)
    g4 = LeakyReLU(alpha = 0.2)(g4)

    g = Conv2D(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g4)
    g = residual_block(g, 512)
    g = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization()(g)
    g = Dropout(drop)(g, training = True)
    g = LeakyReLU(alpha = 0.2)(g)

    g5 = Conv2DTranspose(512, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g)
    g5 = Add()([g4, g5])
    g5 = residual_block(g5, 512)
    g5 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = init)(g5)
    g5 = InstanceNormalization()(g5)
    g5 = LeakyReLU(alpha = 0.2)(g5)

    g6 = Conv2DTranspose(256, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g5)
    g6 = Add()([g3, g6])
    g6 = residual_block(g6, 256)
    g6 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = init)(g6)
    g6 = InstanceNormalization()(g6)
    g6 = LeakyReLU(alpha = 0.2)(g6)

    g7 = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g6)
    g7 = Add()([g2, g7])
    g7 = residual_block(g7, 128)
    g7 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = init)(g7)
    g7 = InstanceNormalization()(g7)

    g7 = LeakyReLU(alpha = 0.2)(g7)

    g8 = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(g7)
    g8 = Add()([g1, g8])
    g8 = residual_block(g8, 64)
    g8 = Conv2D(1, (3,3), padding = 'same', kernel_initializer = init)(g8)
    g8 = InstanceNormalization()(g8)
    g8 = Activation('tanh')(g8)

    model = Model(input_img, g8)

    return model


if __name__ == '__main__':

    model = build_generator()
    model.summary()
