import cv2
import sklearn
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generateZoom(imgs, labs, samples):

    rawGen = []
    labelGen = []
    n = imgs.shape[0]

    for j in range(n):

        img = imgs[j]
        lab = labs[j]
        seed = np.random.randint(1000)
        sampleimg = np.expand_dims(img, 0)
        samplelab = np.expand_dims(lab, 0)
        datagen = ImageDataGenerator(zoom_range = [0.5, 1.0])
        itimg = datagen.flow(sampleimg, batch_size = 1, seed = seed)
        itlabel = datagen.flow(samplelab, batch_size = 1, seed = seed)

        for i in range(samples):

            batchraw = itimg.next()
            imageraw = batchraw[0].astype('uint8')
            batchlabel = itlabel.next()
            imagelabel = batchlabel[0].astype('uint8')
            rawGen.append(imageraw)
            labelGen.append(imagelabel)

    rawGen = np.array(rawGen)
    labelGen = np.array(labelGen)

    return rawGen, labelGen

def shuffle(raw, label):

	raw_shuffled, label_shuffled = sklearn.utils.shuffle(raw, label)

	return raw_shuffled, label_shuffled

def flipInd(imgOrig):

    img = imgOrig.reshape(imgOrig.shape[0], imgOrig.shape[1], 1)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(horizontal_flip = True)
    it = datagen.flow(samples, batch_size = 1)

    while True:
        batch = it.next()
        image = batch[0].astype('uint8')

        if not np.array_equal(data, image):
            break

    return image.reshape(imgOrig.shape)

def genFlip(img, lab):

    imgGen = flipInd(img)
    labGen = flipInd(lab)

    return imgGen, labGen

def flip(raw, lab, save = False, show = False):

    rawGen = np.zeros(raw.shape)
    labGen = np.zeros(lab.shape)

    for i in range(raw.shape[0]):

        rawGen[i], labGen[i] = genFlip(raw[i], lab[i])
        nameR = 'flip/flipRaw' + str(i) + '.jpg'
        nameL = 'flip/flipLab' + str(i) + '.jpg'

        if save:
            cv2.imwrite(nameR, rawGen[i])
            cv2.imwrite(nameL, labGen[i])
        if show:
            cv2.imshow(nameR, rawGen[i])
            cv2.imshow(nameL, labGen[i])
            cv2.waitKey()

    return rawGen, labGen

def zoom(raw, lab, samples, save = False, show = False):

    raw = raw.reshape(raw.shape[0], raw.shape[1], raw.shape[2], 1)
    lab = lab.reshape(lab.shape[0], lab.shape[1], lab.shape[2], 1)
    rawGen, labGen = generateZoom(raw, lab, samples)

    for i in range(rawGen.shape[0]):

        nameR = 'zoom/zoomRaw' + str(i) + '.jpg'
        nameL = 'zoom/zoomLab' + str(i) + '.jpg'

        if save:
            cv2.imwrite(nameR, rawGen[i])
            cv2.imwrite(nameL, labGen[i])
        if show:
            cv2.imshow(nameR, rawGen[i])
            cv2.imshow(nameL, labGen[i])
            cv2.waitKey()

    return rawGen, labGen

def deNormalize(x_in, y_in, u = 128):
    x = x_in.copy()
    y = y_in.copy()
    x = (x * (255/2)) + (255/2)
    y = (y * (255/2)) + (255/2)
    x[x < u] = 0
    x[x >= u] = 1
    y[y < u] = 0
    y[y >= u] = 1

    return x, y
