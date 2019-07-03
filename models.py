#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__: Qmh
# __file_name__: models.py
# __time__: 2019:06:27:19:51

import keras.backend as K
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout,Activation
from keras.layers import Dense,Lambda,Add,GlobalAveragePooling2D,ZeroPadding2D
from keras.regularizers import l2
from keras import Model
import constants as c
from keras.layers.core import Permute
from keras import regularizers

# Resblcok
def res_conv_block(x,filters,strides,name):
    filter1,filer2,filter3 = filters
    # block a
    x = Conv2D(filter1,(1,1),strides=strides,name=f'{name}_conva')(x)
    x = BatchNormalization(name=f'{name}_bna')(x)
    x = Activation('relu',name=f'{name}_relua')(x)
    # block b
    x = Conv2D(filer2,(3,3),padding='same',name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bnb')(x)
    x = Activation('relu',name=f'{name}_relub')(x)
    # block c
    x = Conv2D(filter3,(1,1),name=f'{name}_convc')(x)
    x = BatchNormalization(name=f'{name}_bnc')(x)
    # shortcut
    shortcut = Conv2D(filter3,(1,1),strides=strides,name=f'{name}_shcut')(x)
    shortcut = BatchNormalization(name=f'{name}_stbn')(x)
    x = Add(name=f'{name}_add')([x,shortcut])
    x = Activation('relu',name=f'{name}_relu')(x)
    return x

# ResNet
def ResNet(input_shape):
    x_in = Input(input_shape,name='input')
    # 帧数 特征数  通道数
    # x = Permute((2,1,3),name='permute')(x_in)

    x = Conv2D(64,(5,5),strides=(2,2),padding='same',name='conv1')(x_in)
    x = BatchNormalization(name="bn1")(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2,2),strides=(2,2),padding='same',name='pool1')(x)

    x = res_conv_block(x,(64,64,256),(1,1),name='block1')
    x = res_conv_block(x,(64,64,256),(2,2),name='block2')
    x = res_conv_block(x,(64,64,256),(2,2),name='block3')

    x = res_conv_block(x,(128,128,512),(1,1),name='block4')
    x = res_conv_block(x,(128,128,512),(2,2),name='block5')
    x = res_conv_block(x,(128,128,512),(2,2),name='block6')
    # x = res_conv_block(x,(128,128,512),(2,2),name='block5')
    # avgpool
    # x = GlobalAveragePooling2D(name='avgPool')(x)
    x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)
    # fc2
    x = Dense(512,name='fc2',activation='relu')(x)
    x = BatchNormalization(name='fc_norm')(x)
    x = Activation('relu',name='fc_relu')(x)
    x = Dropout(0.2,name='final_drop')(x)

    model = Model(inputs=[x_in],outputs=[x],name='ResCNN')
    # model.summary()
    return model

def conv_pool(x,layerid,filters,kernal_size,conv_strides,pool_size=None,pool_strides=None,pool=None):
    x = Conv2D(filters,kernal_size,strides= conv_strides,padding='same',name=f'conv{layerid}')(x)
    x = BatchNormalization(name=f'bn{layerid}')(x)
    x = Activation('relu',name=f'relu{layerid}')(x)
    if pool == 'max':
        x = MaxPool2D(pool_size,pool_strides,name=f'mpool{layerid}')(x)
    return x

# vggvox1
def vggvox1_cnn(input_shape):
    x_in = Input(input_shape,name='input')
    x = conv_pool(x_in,1,96,(7,7),(2,2),(3,3),(2,2),'max')
    x = conv_pool(x,2,256,(5,5),(2,2),(3,3),(2,2),'max')
    x = conv_pool(x,3,384,(3,3),(1,1))
    x = conv_pool(x,4,256,(3,3),(1,1))
    x = conv_pool(x,5,256,(3,3),(1,1),(5,3),(3,2),'max')
    # fc 6
    x = Conv2D(512,(9,1),name='fc6')(x)
    # apool6
    x = GlobalAveragePooling2D(name='avgPool')(x)
    # fc7
    x = Dense(512,name='fc7',activation='relu')(x)
    model = Model(inputs=[x_in],outputs=[x],name='vggvox1_cnn')
    return model

# deep speaker
def clipped_relu(inputs):
    return Lambda(lambda y:K.minimum(K.maximum(y,0),20))(inputs)

def identity_block(x_in,kernel_size,filters,name):
    x = Conv2D(filters,kernel_size=kernel_size,strides=(1,1),
    padding='same',kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
    name=f'{name}_conva')(x_in)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = clipped_relu(x)
    x = Conv2D(filters,kernel_size=kernel_size,strides=(1,1),
    padding='same',kernel_regularizer = regularizers.l2(l=c.WEIGHT_DECAY),
    name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Add(name=f'{name}_add')([x,x_in])
    x = clipped_relu(x)
    return x

def Deep_speaker_model(input_shape):

    def conv_and_res_block(x_in,filters):
        x = Conv2D(filters,kernel_size=(5,5),strides=(2,2),
        padding='same',kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
        name=f'conv_{filters}-s')(x_in)
        x = BatchNormalization(name=f'conv_{filters}-s_bn')(x)
        x = clipped_relu(x)
        for i in range(3):
            x = identity_block(x,kernel_size=(3,3),filters=filters,name=f'res{filters}_{i}')
        return x
    
    x_in = Input(input_shape,name='input')
    x = conv_and_res_block(x_in,64)
    x = conv_and_res_block(x,128)
    x = conv_and_res_block(x,256)
    x = conv_and_res_block(x,512)
    # average
    x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)
    # affine
    x = Dense(512,name='affine')(x)
    x = Lambda(lambda y:K.l2_normalize(y,axis=1),name='ln')(x)
    model = Model(inputs=[x_in],outputs=[x],name='deepspeaker')
    return model


if __name__ == "__main__":
    
    # model = ResNet(c.INPUT_SHPE)
    # model = vggvox1_cnn((512,299,1))
    model = Deep_speaker_model(c.INPUT_SHPE)
    print(model.summary())
   