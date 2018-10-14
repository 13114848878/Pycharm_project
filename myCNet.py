#搭建卷积神经网络，目的为了实现用脑电预测被试疲劳水平。使用kearas搭建
#date：2018-10-14
#Author：Sen Tian
#email：414319563@qq.com
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K



print('OK')

def CNeTS(shape,conv_layer_number):


# class CNeTS:
#
#
#     def __init__(self,shape):



