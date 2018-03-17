import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Deconv2D
from keras.applications.vgg16 import VGG16
from keras.layers.merge import Add

def FCN_VGG16_32s(input_shape, class_num):

    # input
    input = Input(shape=input_shape) # (h, w, c)
    
    # block1
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)

    # block2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)

    # block3
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)

    # block4
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)

    # block5
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)

    # fc (implemented as conv)
    x = Conv2D(4096, (7,7), activation='relu', padding='same', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(class_num, (1,1), name='fc3')(x) # No activation (i.e. a(x) = x)

    # upsampling (x32)
    x = Deconv2D(class_num, (32, 32), strides=(32, 32), name='deconv', use_bias=False, activation='softmax')(x) # padding?
    
    # define model
    model = Model(input, x)

    return model


def FCN_VGG16_32s_rgbd(input_shape, class_num):
    # input
    input = Input(shape=input_shape) # (h, w, c)
    
    # VGG16
    vgg16 = VGG16(input_shape=(input_shape[0], input_shape[1], 3), include_top=False, weights='imagenet')

    # first layer
    if input_shape[2] == 3: # RGB
        x = vgg16(input)
    
    elif input_shape[2] == 4: # RGBD
        # split input into RGB and D channels
        rgb = Lambda(lambda x : x[:,:,:,0:3])(input)
        depth = Lambda(lambda x : x[:,:,:,3:4])(input)

        # first layer
        rgb_feat = vgg16.layers[1](rgb) # block1_conv1
        depth_feat = Conv2D(64, (3,3), activation='relu', padding='same', name='depth_conv')(depth)
        x = Add()([rgb_feat, depth_feat])
        for i in range(2, len(vgg16.layers)):
            x = vgg16.layers[i](x)
    
    else:
        print("Error!! wrong # of channels!!")
        sys.exit(1)

    # fc (implemented as conv)
    x = Conv2D(4096, (7,7), activation='relu', padding='same', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(class_num, (1,1), name='fc3')(x) # No activation (i.e. a(x) = x)

    # upsampling (x32)
    x = Deconv2D(class_num, (32, 32), strides=(32, 32), name='deconv', use_bias=False, activation='softmax')(x) # padding?
    
    # define model
    model = Model(input, x)

    return model 