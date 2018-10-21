#搭建卷积神经网络，目的为了实现用脑电预测被试疲劳水平。使用kearas搭建
#date：2018-10-14
#Author：Sen Tian
#email：414319563@qq.com
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import SpatialDropout2D
# from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

print('OK')

path = 'D:\\data\\EEG\\newdata\\'


shape = [32,32]
batch_size = 16
is_train = True
filters_list = [8,16] #output channel
fc_units_list = [256,128]
rate = 0 # drop rate
X = Input(shape=shape, batch_size=batch_size, name='Input', dtype=tf.float32,)
a = BatchNormalization(momentum=0.999, trainable=is_train, name='BN')(X)


def conv_pool(X,filters,is_train):


    #kernel_size = [3, 3]  # [3,3]/3,

    z = Conv2D(filters=filters, padding='same', kernel_size=[3,3],
               data_format='channels_last', use_bias=False)(X)
    z = BatchNormalization(momentum=0.999, trainable=is_train, name='BN')(z)
    a = Activation('relu')(z)
    a = MaxPooling2D()(a)
    return a

#convolution layers
for filters in filters_list:
    a = conv_pool(a, filters, is_train)

a = Flatten(name = 'flatten')(a) #batch_number,feature
# full connection layers
for units in fc_units_list:
    z = Dense(units=units, use_bias=False, )(a)
    z = BatchNormalization(momentum=0.999, trainable=is_train, name='BN')(z)
    a = Activation('relu')(z)
    a = Dropout(rate, )(a)

z = Dense(units=Y.shape[1], use_bias=False, )(a)
z = BatchNormalization(momentum=0.999, trainable=is_train, name='BN')(z)
Y_pre = Activation('softmax', name='softmax')(z)

Model(inputs=X, outputs=Y_pre)

lr = 0.001
optimizer = Adam(lr=lr)
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y_batch, logits=Y_pre)
Model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
callbacks = EarlyStopping(patience=50)
Model.fit(x=X_train, y=Y_train, batch_size=batch_size, callbacks=callbacks,
          validation_data=(X_val,Y_val),)

loss_val,acc_val = Model.evaluate(x=X_val, y=Y_val, batch_size=batch_size, )

# def CNeTS(shape,conv_layer_number):
#

# class CNeTS:
#
#
#     def __init__(self,shape):



