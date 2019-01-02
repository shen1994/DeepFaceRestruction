# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:24:57 2018

@author: shen1994
"""

import cv2
import numpy as np
from utils.cv_plot import plot_kpt

class Predictor:
    
    def __init__(self, 
                 in_shape=(256, 256, 3),
                 out_shape=(256, 256, 3),
                 texture_size=256):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.texture_size = texture_size
        self.uv_kpt_ind = np.loadtxt('images/uv_kpt_ind.txt').astype(np.int32)
        self.face_ind = np.loadtxt("images/face_ind.txt").astype(np.int32)
        self.triangles = np.loadtxt("images/triangles.txt").astype(np.int32)
        self.uv_coords = self.generate_uv_coords()
        
    def generate_uv_coords(self):
        resolution = self.out_shape[0]
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1]);
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        
        return uv_coords

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.out_shape[0]**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices
        
    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors
        
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt

    def predictor(self, sess, x, y, image):
        
        # run model to get uv_map
        pos = sess.run(y, feed_dict={x: image[np.newaxis, :,:,:]})
        pos = np.squeeze(pos)
        max_pos = self.in_shape[0]
        pos = pos * max_pos

        t_image = (image*255.).astype(np.uint8)
        kpt = self.get_landmarks(pos)
        kpt_origin = plot_kpt(image, kpt).astype(np.uint8)
        kpt_gray = cv2.cvtColor(kpt_origin, cv2.COLOR_RGB2GRAY) 
        ret, kpt_mask = cv2.threshold(kpt_gray, 127, 255, cv2.THRESH_BINARY) 
        kpt_mask = cv2.bitwise_not(kpt_mask)
        kpt_and = cv2.bitwise_and(t_image, t_image, mask=kpt_mask)
        kpt_image = cv2.add(kpt_and, kpt_origin)
        
        return kpt_image / 255.

            