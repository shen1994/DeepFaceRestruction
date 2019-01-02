# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:10:11 2018

@author: shen1994
"""

import os
import cv2
import numpy as np

from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from utils.render_app import get_visibility
from utils.render_app import get_uv_mask
from utils.write import write_obj_with_texture
from utils.write import write_obj_with_colors
from utils.cv_plot import plot_kpt
    
def get_vertices(pos, face_ind, resolution_op):

    all_vertices = np.reshape(pos, [resolution_op**2, -1])
    vertices = all_vertices[face_ind, :]

    return vertices
        
def get_colors(image, vertices):

    [h, w, _] = image.shape
    vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)
    vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:,1], ind[:,0], :]
    
    return colors
        
def get_landmarks(pos, uv_kpt_ind):

    kpt = pos[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
    return kpt
    
def generate_uv_coords(face_ind, resolution_op):
    uv_coords = np.meshgrid(range(resolution_op),range(resolution_op))
    uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
    uv_coords = np.reshape(uv_coords, [resolution_op**2, -1]);
    uv_coords = uv_coords[face_ind, :]
    uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
            
    return uv_coords

def run_one_image(uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  image_path, npy_path, save_folder, name, 
                  uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
    
    # 1. load image
    cropped_image = imread(image_path) / 255.
    
    print('input image is ok!')

    # 2. load uv position map
    pos = np.load(npy_path)
    
    print('uv map is ok!')
    
    # 3. deal uv map
    # run model to get uv_map
    max_pos = image_h
    pos = pos * max_pos

    # 4. get useful vertices
    vertices = get_vertices(pos, face_ind, uv_h)
    save_vertices = vertices.copy()
    save_vertices[:,1] = image_h - 1 - save_vertices[:,1]

    # 5. get colors
    colors = get_colors(cropped_image, vertices)
    write_obj_with_colors(os.path.join(save_folder, name + '_c.obj'), 
                          save_vertices, triangles, colors)
    
    print('color 3d face is ok!')

    # 6. get texture
    pos_interpolated = pos.copy()
    texture = cv2.remap(cropped_image, pos_interpolated[:,:,:2].astype(np.float32), 
                        None, interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    vertices_vis = get_visibility(vertices, triangles, image_h, image_w)
    uv_mask = get_uv_mask(vertices_vis, triangles, s_uv_coords, image_h, image_w, uv_h)
    uv_mask = resize(uv_mask, (256, 256), preserve_range = True)
    texture = texture * uv_mask[:, :, np.newaxis]
    write_obj_with_texture(os.path.join(save_folder, name + '.obj'), 
                           save_vertices, triangles, 
                           texture, s_uv_coords/uv_h)
    
    print('texture 3d face is ok!')
 
    # 7. get landmarks
    t_image = (cropped_image*255.).astype(np.uint8)
    kpt = get_landmarks(pos, uv_kpt_ind)
    kpt_origin = plot_kpt(cropped_image, kpt).astype(np.uint8)
    kpt_gray = cv2.cvtColor(kpt_origin, cv2.COLOR_RGB2GRAY) 
    ret, kpt_mask = cv2.threshold(kpt_gray, 127, 255, cv2.THRESH_BINARY) 
    kpt_mask = cv2.bitwise_not(kpt_mask)
    kpt_and = cv2.bitwise_and(t_image, t_image, mask=kpt_mask)
    kpt_image = cv2.add(kpt_and, kpt_origin)
    imsave(os.path.join(save_folder, name + '_kpt.jpg'), kpt_image / 255.)
    
    print('kpt image is ok!')


if __name__ == "__main__":

    uv_kpt_ind = np.loadtxt('images/uv_kpt_ind.txt').astype(np.int32)
    face_ind = np.loadtxt("images/face_ind.txt").astype(np.int32)
    triangles = np.loadtxt("images/triangles.txt").astype(np.int32)
    s_uv_coords = generate_uv_coords(face_ind, 256)
    
    print('load location ok!')
    
    image_path = 'images/300W_LP/AFW_GEN/AFW_111076519_1_1.jpg'
    mat_path = 'images/300W_LP/AFW_GEN/AFW_111076519_1_1.npy'
   
    run_one_image(uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  image_path, mat_path, 'images', 'test_g')
    