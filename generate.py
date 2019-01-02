# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:48:43 2018

@author: shen1994
"""

import os
import cv2
import glob
import numpy as np
from skimage.io import imread

class Generator(object):
    
    def __init__(self,
        train_paths,
        mask_path, 
        valid_path,
        image_shape = (256, 256, 3),
        batch_size = 16):
        
        train_images, train_labels = [], []
        for head_path in train_paths:
            image_paths = glob.glob(os.path.join(head_path, '*.jpg'))
            label_paths = glob.glob(os.path.join(head_path, '*.npy'))
            image_paths = [image_path.replace('\\', '/') for image_path in image_paths]
            label_paths = [label_path.replace('\\', '/') for label_path in label_paths]
            for image_path in image_paths:
                image_name = image_path.split('/')[-1][:-4]
                npy_path = head_path + '/' + image_name + '.npy'
                if npy_path in label_paths:
                    train_images.append(image_path)
                    train_labels.append(npy_path)
                    
        valid_images = glob.glob(os.path.join(valid_path, '*.jpg'))
        
        self.mask = cv2.imread(mask_path, 0).astype('float32')
        
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.train_length = len(self.train_images)
        self.valid_length = len(self.valid_images)
        
    def generate(self, is_training=True):
        while(True):
            if is_training:
                rand_idx = [one for one in range(self.train_length)]
                np.random.shuffle(rand_idx)
                self.train_images = [self.train_images[one] for one in rand_idx]
                self.train_labels = [self.train_labels[one] for one in rand_idx]

                counter = 0
                x_array, y_array, m_array = [], [], []
                for index in range(self.train_length):
                    image = imread(self.train_images[index]) / 255.
                    label = np.load(self.train_labels[index])
                    
                    x_array.append(image)
                    y_array.append(label)
                    m_array.append(self.mask)
                    counter += 1
                    
                    if counter >= self.batch_size:
                        yield (np.array(x_array), np.array(y_array), np.array(m_array))
                        counter = 0
                        x_array, y_array, m_array = [], [], []
            else:
                rand_idx = [one for one in range(self.valid_length)]
                np.random.shuffle(rand_idx)
                self.valid_images = [self.valid_images[one] for one in rand_idx]

                counter = 0; i_array = []
                for index in range(self.valid_length):
                    image = imread(self.valid_images[index]) / 255.
                    i_array.append(image)
                    
                    counter += 1
                    if counter >= self.batch_size:
                        yield i_array
                        counter = 0; i_array = []
                             