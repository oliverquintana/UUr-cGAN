import os
import numpy as np
import pandas as pd
from helper import *
from generator import *
from discriminator import *
from medpy.metric.binary import dc
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class cGAN():

    def __init__(self, img_shape = [256, 256, 1], gModel = None, dModel = None, ganModel = None, save_path = ''):

        self.dModel = dModel
        self.gModel = gModel
        self.ganModel = ganModel
        self.img_shape = img_shape
        self.save_path = save_path

        try:
            os.mkdir(self.save_path + 'saved_model')
        except:
            os.mkdir(self.save_path)
            os.mkdir(self.save_path + 'saved_model')

    def build(self, lr = 0.0002, b = [0.9, 0.999], dropG = 0.0, dropD = 0.0, dModel = None, gModel = None, lossWeights = [1, 75]):

        if dModel == None:
	        self.dModel = build_discriminator(self.img_shape, lr, b, dropD)
        else:
            self.dModel = load_model(dModel)

        if gModel == None:
            self.gModel = build_generator(self.img_shape, dropG)
        else:
            self.gModel = load_model(gModel)

        self.dModel.trainable = False
        input_img = Input(self.img_shape)
        gen_out = self.gModel(input_img)
        dis_out = self.dModel([input_img, gen_out])
        self.ganModel = Model(input_img, [dis_out, gen_out])

        lr_schedule = ExponentialDecay(
            initial_learning_rate = lr,
            decay_steps = 1000,
            decay_rate = 0.9)
        opt = Adam(learning_rate = lr_schedule, beta_1 = b[0])#, clipvalue = 1.0)

        self.ganModel.compile(loss = ['binary_crossentropy', 'mse'], optimizer = opt, loss_weights = lossWeights)

    def train(self, dataset, val_dataset, n_epochs = 10, n_batch = 1, n_patch = 16):

        def generate_real_samples(dataset, n_samples, patch_shape):
            trainA, trainB = dataset
            ix = np.random.randint(0, trainA.shape[0], n_samples)
            x1, x2 = trainA[ix], trainB[ix]
            y = np.random.uniform(.7, 1.2, (n_samples, patch_shape, patch_shape, 1))
            return [x1, x2], y

        def generate_fake_samples(samples, patch_shape):
            x = self.gModel.predict(samples)
            y = np.random.uniform(0, .3, (len(x), patch_shape, patch_shape, 1))
            return x, y

        x_test, y_test = val_dataset
        trainA, trainB = dataset
        bat_per_epo = int(len(trainA) / n_batch)
        n_steps = bat_per_epo * n_epochs
        dc_prev = 0

        for i in range(n_steps):
            [xRealA, xRealB], yReal = generate_real_samples(dataset, n_batch, n_patch)
            xFakeB, yFake = generate_fake_samples(xRealA, n_patch)
            d_loss1 = self.dModel.train_on_batch([xRealA, xRealB], yReal)
            d_loss2 = self.dModel.train_on_batch([xRealA, xFakeB], yFake)
            d_loss = 0.5 * (d_loss1 + d_loss2)
            g_loss = self.ganModel.train_on_batch(xRealA, [yReal, xRealB])

            if ((i / bat_per_epo) % 1) == 0 or i == n_steps-1:
                print('Epoch: {}'.format(i / bat_per_epo))

                x = self.gModel.predict(trainA)
                x, y = deNormalize(x, trainB)

                dc_train = dc(x, y)
                print('DSC Train: {}'.format(dc_train))

                x = self.gModel.predict(x_test)
                x, y = deNormalize(x, y_test)
                dc_test = dc(x, y)

                testDSC.append(dc_test)
                print('DSC Test: {}'.format(dc_test))

                dModelHist.append(d_loss)
                gModelHist.append(g_loss[0])
                dloss1_hist.append(d_loss1)
                dloss2_hist.append(d_loss2)

                if dc_test > dc_prev:
                    path = 'saved_model/'
                    try:
                        os.mkdir(path)
                    except:
                        pass
                    self.dModel.save(self.save_path + '/saved_model/dModel.h5', overwrite = True)
                    self.gModel.save(self.save_path + '/saved_model/gModel.h5', overwrite = True)
                    dc_prev = dc_test


if __name__ == '__main__':

    model = cGAN()
    model.build()
    model.ganModel.summary()
