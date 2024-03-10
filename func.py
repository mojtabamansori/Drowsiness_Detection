import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.stats as stats
from tensorflow.keras .utils import to_categorical
import matplotlib.pyplot as plt

fp1, fp2, f7 = 0, 1, 2
f3, fz, f4 = 3, 4, 5
f8, ft7, fc3 = 6, 7, 8
fcz, fc4, ft8 = 9, 10, 11
t3, c3, cz = 12, 13, 14
c4, t4, tp7 = 15, 16, 17
cp3, cpz, cp4 = 18, 19, 20
tp8a1, t5, p3 = 21, 22, 23
pz, p4, t6a2 = 24, 25, 26
o1, oz, o2 = 27, 28, 29
nepoch = 35
basize = 256
lr = 0.0001

startsub = np.zeros(11)
finalsub = np.zeros(11)
startsub[0] = 0
finalsub[0] = 188
startsub[1] = finalsub[0]
finalsub[1] = 320
startsub[2] = finalsub[1]
finalsub[2] = 470
startsub[3] = finalsub[2]
finalsub[3] = 618
startsub[4] = finalsub[3]
finalsub[4] = 842
startsub[5] = finalsub[4]
finalsub[5] = 1008
startsub[6] = finalsub[5]
finalsub[6] = 1110
startsub[7] = finalsub[6]
finalsub[7] = 1374
startsub[8] = finalsub[7]
finalsub[8] = 1688
startsub[9] = finalsub[8]
finalsub[9] = 1796
startsub[10] = finalsub[9]
finalsub[10] = 2022


def bipolar(xdata):
    xdatabipolar = np.zeros((2022, 32, 384))
    # amodi
    # 1
    xdatabipolar[:, 0, :] = xdata[:, fp1, :] - xdata[:, f7, :]
    xdatabipolar[:, 1, :] = xdata[:, f7, :] - xdata[:, t3, :]
    xdatabipolar[:, 2, :] = xdata[:, t3, :] - xdata[:, t5, :]
    xdatabipolar[:, 3, :] = xdata[:, t5, :] - xdata[:, o1, :]
    xdatabipolar[:, 4, :] = xdata[:, o1, :] - xdata[:, p3, :]
    xdatabipolar[:, 5, :] = xdata[:, p3, :] - xdata[:, c3, :]
    xdatabipolar[:, 6, :] = xdata[:, c3, :] - xdata[:, f3, :]
    xdatabipolar[:, 7, :] = xdata[:, f3, :] - xdata[:, fp1, :]
    # 2
    xdatabipolar[:, 8, :] = xdata[:, pz, :] - xdata[:, cz, :]
    xdatabipolar[:, 9, :] = xdata[:, cz, :] - xdata[:, fz, :]
    # 3
    xdatabipolar[:, 10, :] = xdata[:, fp2, :] - xdata[:, f8, :]
    xdatabipolar[:, 11, :] = xdata[:, f8, :] - xdata[:, t4, :]
    xdatabipolar[:, 12, :] = xdata[:, t4, :] - xdata[:, t6a2, :]
    xdatabipolar[:, 13, :] = xdata[:, t6a2, :] - xdata[:, o2, :]
    xdatabipolar[:, 14, :] = xdata[:, o2, :] - xdata[:, p4, :]
    xdatabipolar[:, 15, :] = xdata[:, p4, :] - xdata[:, c4, :]
    xdatabipolar[:, 16, :] = xdata[:, c4, :] - xdata[:, f4, :]
    xdatabipolar[:, 17, :] = xdata[:, f4, :] - xdata[:, fp2, :]
    # ofghi
    # 4
    xdatabipolar[:, 18, :] = xdata[:, fp1, :] - xdata[:, fp2, :]
    xdatabipolar[:, 19, :] = xdata[:, f8, :] - xdata[:, f4, :]
    xdatabipolar[:, 20, :] = xdata[:, f4, :] - xdata[:, fz, :]
    xdatabipolar[:, 21, :] = xdata[:, fz, :] - xdata[:, f3, :]
    xdatabipolar[:, 22, :] = xdata[:, f3, :] - xdata[:, f7, :]
    # 5
    xdatabipolar[:, 23, :] = xdata[:, t3, :] - xdata[:, c3, :]
    xdatabipolar[:, 24, :] = xdata[:, c3, :] - xdata[:, cz, :]
    xdatabipolar[:, 25, :] = xdata[:, cz, :] - xdata[:, c4, :]
    xdatabipolar[:, 26, :] = xdata[:, c4, :] - xdata[:, t4, :]
    # 6
    xdatabipolar[:, 27, :] = xdata[:, t5, :] - xdata[:, p3, :]
    xdatabipolar[:, 28, :] = xdata[:, p3, :] - xdata[:, pz, :]
    xdatabipolar[:, 29, :] = xdata[:, pz, :] - xdata[:, p4, :]
    xdatabipolar[:, 30, :] = xdata[:, p4, :] - xdata[:, t6a2, :]
    xdatabipolar[:, 31, :] = xdata[:, o2, :] - xdata[:, o1, :]
    return xdatabipolar


def zscoresubjective(xdatabipolar):
    startsub = np.zeros(11)
    finalsub = np.zeros(11)
    startsub[0] = 0
    finalsub[0] = 188 * 3 - 2
    startsub[1] = finalsub[0]
    finalsub[1] = 320 * 3 - 2
    startsub[2] = finalsub[1]
    finalsub[2] = 470 * 3 - 2
    startsub[3] = finalsub[2]
    finalsub[3] = 618 * 3 - 2
    startsub[4] = finalsub[3]
    finalsub[4] = 842 * 3 - 2
    startsub[5] = finalsub[4]
    finalsub[5] = 1008 * 3 - 2
    startsub[6] = finalsub[5]
    finalsub[6] = 1110 * 3 - 2
    startsub[7] = finalsub[6]
    finalsub[7] = 1374 * 3 - 2
    startsub[8] = finalsub[7]
    finalsub[8] = 1688 * 3 - 2
    startsub[9] = finalsub[8]
    finalsub[9] = 1796 * 3 - 2
    startsub[10] = finalsub[9]
    finalsub[10] = 2022 * 3 - 2

    for i in range(0, 11):
        xdatabipolar[int(startsub[i]):int(finalsub[i])] = stats.zscore(xdatabipolar[int(startsub[i]):int(finalsub[i])])
    return xdatabipolar

def hamposhani(xdata,label):
    newdata = np.zeros((6063, 30, 384))
    for i in range(2021):
        newdata[(3 * i), :, :] = xdata[i, :, :]

        newdata[(3 * i) + 1, :, :256] = xdata[i, :, 128:]
        newdata[(3 * i) + 1, :, 256:] = xdata[i + 1, :, :128]

        newdata[(3 * i) + 2, :, :128] = xdata[i, :, 256:]
        newdata[(3 * i) + 2, :, 128:] = xdata[i + 1, :, :256]

    newlabel = np.zeros((6063))
    for i in range(2021):
        newlabel[(3 * i)] = label[i]
        newlabel[(3 * i) + 1] = label[i]
        newlabel[(3 * i) + 2] = label[i + 1]
    return newdata,newlabel

def hamposhanisub(xdata,label,ncha):

    n = len(xdata)
    k = 3 * n - 2
    newdata = np.zeros((k, ncha, 384))
    newlabel = np.zeros((k))
    leni1 = 0
    for iindex in range(11):
        s = startsub[iindex]
        f = finalsub[iindex]
        leni2 = leni1 + int(f - s)
        for i in range(leni1, leni2-1 ):
            newdata[(3 * i), :, :] = xdata[i, :, :]

            newdata[(3 * i) + 1, :, :256] = xdata[i, :, 128:]
            newdata[(3 * i) + 1, :, 256:] = xdata[i + 1, :, :128]

            newdata[(3 * i) + 2, :, :128] = xdata[i, :, 256:]
            newdata[(3 * i) + 2, :, 128:] = xdata[i + 1, :, :256]

        for i in range(leni1, leni2 - 1):
            newlabel[(3 * i)] = label[i]
            newlabel[(3 * i) + 1] = label[i]
            newlabel[(3 * i) + 2] = label[i + 1]
        leni1 = leni2
    startsub[0] = 0
    finalsub[0] = 188 * 3 - 2
    startsub[1] = finalsub[0]
    finalsub[1] = 320 * 3 - 2
    startsub[2] = finalsub[1]
    finalsub[2] = 470 * 3 - 2
    startsub[3] = finalsub[2]
    finalsub[3] = 618 * 3 - 2
    startsub[4] = finalsub[3]
    finalsub[4] = 842 * 3 - 2
    startsub[5] = finalsub[4]
    finalsub[5] = 1008 * 3 - 2
    startsub[6] = finalsub[5]
    finalsub[6] = 1110 * 3 - 2
    startsub[7] = finalsub[6]
    finalsub[7] = 1374 * 3 - 2
    startsub[8] = finalsub[7]
    finalsub[8] = 1688 * 3 - 2
    startsub[9] = finalsub[8]
    finalsub[9] = 1796 * 3 - 2
    startsub[10] = finalsub[9]
    finalsub[10] = 2022 * 3 - 2
    return newdata,newlabel,finalsub, startsub

class Model(tf.keras.Model):
    def __init__(self, N1, sampleChannel, d, kernelLength, sampleLength, classes):
        super(Model, self).__init__()
        self.pointwise = tf.keras.layers.Conv2D(N1, (sampleChannel, 1))
        self.depthwise = tf.keras.layers.DepthwiseConv2D((1, kernelLength), depth_multiplier=d, padding='valid',
                                                         depthwise_initializer='glorot_uniform')
        self.activ = tf.keras.layers.ReLU()
        self.batchnorm = tf.keras.layers.BatchNormalization(scale=False, center=True, trainable=True)
        self.GAP = tf.keras.layers.AveragePooling2D(pool_size=(1, sampleLength - kernelLength + 1))
        self.dropout = tf.keras.layers.GaussianDropout(0.3)
        self.fc = tf.keras.layers.Dense(classes)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, inputs):
        x = self.pointwise(inputs)
        x = self.depthwise(x)
        x = self.activ(x)
        x = self.batchnorm(x)
        x = self.GAP(x)
        x = tf.squeeze(x, axis=[1, 2])
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x