# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:10:11 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
import scipy.io as sio
import skimage.transform
from skimage import io

from skimage.io import imread
from skimage.io import imsave
from utils.render import render_texture

from mm3d import mesh
from mm3d.morphable_model import MorphabelModel
from mm3d.morphable_model.load import load_uv_coords

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords
    
def get_vertices(pos, face_ind, resolution_op):

    all_vertices = np.reshape(pos, [resolution_op**2, -1])
    vertices = all_vertices[face_ind, :]

    return vertices
    
def get_colors_from_texture(texture, face_ind, resolution_op):

    all_colors = np.reshape(texture, [resolution_op**2, -1])
    colors = all_colors[face_ind, :]

    return colors
        
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
    
def run_one_image(bfm, uv_coords, face_ind,
                  image_path, mat_path, uv_h, uv_w, image_h, image_w):
    
    # 1. load image and fitted parameters
    image = io.imread(image_path)/255.
    [h, w, c] = image.shape

    info = sio.loadmat(mat_path)
    pose_para = info['Pose_Para'].T.astype(np.float32)
    shape_para = info['Shape_Para'].astype(np.float32)
    exp_para = info['Exp_Para'].astype(np.float32)

    # 2. generate mesh
    vertices = bfm.generate_vertices(shape_para, exp_para)
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()
    image_vertices = projected_vertices.copy()
    image_vertices[:,1] = h - image_vertices[:,1] - 1

    # 3. crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0, 
             bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top)/2
    size = int(old_size * 1.5)
    # random pertube. you can change the numbers
    marg = old_size * 0.1
    t_x = np.random.rand() * marg * 2 - marg
    t_y = np.random.rand() * marg * 2 - marg
    center[0] = center[0] + t_x; center[1] = center[1] + t_y
    size = size*(np.random.rand()*0.2 + 0.9)  
    
    # crop and record the transform parameters
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0]-size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))
    
    print('input image is ok!')

    # transform face position(image vertices) along with 2d facial image 
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2]*tform.params[0, 0] # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2]) # translate z

    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c = 3)
    
    print('uv map is ok!')

    # 6. get useful vertices
    vertices = get_vertices(uv_position_map, face_ind, uv_h)
    
    return image, cropped_image, (center[0], center[1]), size, uv_position_map, vertices

def run_two_image(bfm, uv_coords, uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  image_path_A, mat_path_A, image_path_B, mat_path_B, save_folder, name, mode=1,
                  uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
    
    image, cropped_image, center, size, pos, vertices = \
        run_one_image(bfm, uv_coords, face_ind, image_path_A, mat_path_A, 
                  uv_h, uv_w, image_h, image_w)
        
    ref_image, ref_cropped_image, ref_center, ref_size, ref_pos, ref_vertices = \
        run_one_image(bfm, uv_coords, face_ind, image_path_B, mat_path_B, 
                  uv_h, uv_w, image_h, image_w)
    
    texture = cv2.remap(cropped_image, pos[:,:,:2].astype(np.float32), 
                        None, interpolation=cv2.INTER_NEAREST, 
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_texture = cv2.remap(ref_cropped_image, ref_pos[:,:,:2].astype(np.float32), 
                            None, interpolation=cv2.INTER_NEAREST, 
                            borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    if mode == 0: 
        # load eye mask
        uv_face_eye = imread('images/uv_face_eyes.png', as_grey=True) / 255. 
        uv_face = imread('images/uv_face.png', as_grey=True) / 255.
        eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)
        # modify texture
        new_texture = texture*(1 - eye_mask[:,:,np.newaxis]) + ref_texture*eye_mask[:,:,np.newaxis]
    else: 
        uv_whole_face = imread('images/uv_face_mask.png', as_grey=True) / 255.
        new_texture = texture*(1 - uv_whole_face[:,:,np.newaxis]) + ref_texture*uv_whole_face[:,:,np.newaxis]
        # new_texture = ref_texture
            
    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, triangles.T, image_h, image_w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)
    new_colors = get_colors_from_texture(new_texture, face_ind, uv_h)
    new_image = render_texture(vertices.T, new_colors.T, triangles.T, image_h, image_w, c = 3)
    new_image = cropped_image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))

    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (cropped_image*255).astype(np.uint8), 
                               (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    if mode == 0:
        imsave(os.path.join(save_folder, name + '_eyes.jpg'), output)
    else:
        imsave(os.path.join(save_folder, name + '_swap.jpg'), output)

if __name__ == "__main__":

    # load bfm model
    bfm = MorphabelModel('mm3d/BFM/BFM.mat')  
    uv_coords = load_uv_coords('mm3d/BFM/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)
    
    print('load bfm ok!')

    uv_kpt_ind = np.loadtxt('images/uv_kpt_ind.txt').astype(np.int32)
    face_ind = np.loadtxt("images/face_ind.txt").astype(np.int32)
    triangles = np.loadtxt("images/triangles.txt").astype(np.int32)
    s_uv_coords = generate_uv_coords(face_ind, 256)
    
    print('load location ok!')
    
    image_path_A = 'images/300W_LP/AFW/AFW_261068_1_1.jpg'
    mat_path_A = 'images/300W_LP/AFW/AFW_261068_1_1.mat'
    image_path_B = 'images/300W_LP/AFW/AFW_1634816_1_0.jpg'
    mat_path_B = 'images/300W_LP/AFW/AFW_1634816_1_0.mat'
   
    run_two_image(bfm, uv_coords, uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  image_path_A, mat_path_A, image_path_B, mat_path_B, 'images', 'test')
    