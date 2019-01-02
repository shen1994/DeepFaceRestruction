# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:46:30 2018

@author: shen1994
"""

import cv2
import numpy as np
from skimage.transform import resize

def get_transpose_axes(n):
    
    if n % 2 == 0:
        y_axes = list(range( 1, n-1, 2 ))
        x_axes = list(range( 0, n-1, 2 ))
    else:
        y_axes = list(range( 0, n-1, 2 ))
        x_axes = list(range( 1, n-1, 2 ))
        
    return y_axes, x_axes, [n-1]

def stack_images(images):
    
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len( images_shape ))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]

    return np.transpose(images, axes=np.concatenate(new_axes)).reshape( new_shape)
    
def resize_images(images, size):

    new_images = []
    for image in images:
        nimage = resize(image, (size, size), preserve_range = True)
        new_images.append(nimage)
    return np.array(new_images)

def show_G(images_A, images_B, batch_size, name):
    
    images_A = resize_images(images_A, 128)
    images_B = resize_images(images_B, 128)
    figure = np.stack([images_A, images_B], axis=1)
    figure = figure.reshape((4, batch_size//4) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip(figure * 255.0, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    cv2.imshow(name, figure)    
    