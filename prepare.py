# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:13:50 2018

@author: shen1994
"""

import os
import glob
import numpy as np
import scipy.io as sio
import skimage.transform
from skimage import io
from skimage.io import imsave
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import sys
sys.path.append('..')

from mm3d import mesh
from mm3d.morphable_model import MorphabelModel
from mm3d.morphable_model.load import load_uv_coords

def postmap_from_300WLP(bfm, kpt_ind, triangles, uv_coords, image_path, mat_path, 
                        uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
    
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
     # using stantard camera & orth projection as in 3DDFA
    projected_vertices = transformed_vertices.copy()
    image_vertices = projected_vertices.copy()
    image_vertices[:,1] = h - image_vertices[:,1] - 1

    # 3. crop image with key points
    kpt = image_vertices[kpt_ind, :].astype(np.int32)
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
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))

    # transform face position(image vertices) along with 2d facial image 
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2]*tform.params[0, 0] # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2]) # translate z

    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, triangles, position, uv_h, uv_w, c = 3)
    uv_position_map = uv_position_map / max(image_h, image_w)
    
    return cropped_image, uv_position_map
    
def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:, 0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:, 1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords
    
def process_one(number, bfm, kpt_ind, triangles, uv_coords, head_path, save_path, image_paths):
    counter = 0
    print('N%d--->Start!' %(number))
    for image_path in image_paths:
        counter += 1
        image_name = image_path.split('/')[-1][:-4]
        mat_path = head_path + '/' + image_name + '.mat'
        
        cropped_image, uv_position_map = postmap_from_300WLP(bfm, kpt_ind, triangles, uv_coords, image_path, mat_path)
        imsave('{}/{}'.format(save_path, image_name + '.jpg'), cropped_image)
        np.save('{}/{}'.format(save_path, image_name + '.npy'), uv_position_map)
        print('N%d--->Full Samples: %d, Current Counter: %d' %(number, len(image_paths), counter))      

class MyManager(BaseManager):
    pass
MyManager.register('MorphabelModel', MorphabelModel)
def into_manager():
    m = MyManager()
    m.start()
    return m

if __name__ == "__main__":
    
    # 1. load bfm model
    bfm = MorphabelModel('mm3d/BFM/BFM.mat')
    kpt_ind = bfm.kpt_ind
    triangles = bfm.full_triangles
    uv_coords = load_uv_coords('mm3d/BFM/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)
    
    # 2. use the method
    bfm = into_manager().MorphabelModel('mm3d/BFM/BFM.mat')
    
    # 3. create head path
    head_paths = ['images/300W_LP/AFW', 'images/300W_LP/HELEN',
                   'images/300W_LP/IBUG', 'images/300W_LP/LFPW',
                   'images/300W_LP/AFW_Flip', 'images/300W_LP/HELEN_Flip',
                   'images/300W_LP/IBUG_Flip', 'images/300W_LP/LFPW_Flip', 
                   'images/AFLW2000']
                   
    save_paths = []
    for head_path in head_paths:
        save_path = head_path + '_GEN'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_paths.append(save_path)
     
    # 3. check and proceed all files
    # change here, if larger, speed is faster, but need more time!
    base_number = 4000
    base_process = []
    for index, head_path in enumerate(head_paths):
        # glob images
        origin_image_paths = glob.glob(os.path.join(head_path, '*.jpg'))
        origin_image_paths = [image_path.replace('\\', '/') for image_path in origin_image_paths]
                       
        cropped_image_paths = glob.glob(os.path.join(head_path+'_GEN', '*.jpg'))
        cropped_image_paths = [image_path.replace('\\', '/') for image_path in cropped_image_paths]
            
        # avoid repeatted file
        image_paths = []
        for image_path in origin_image_paths:
            if head_path+'_GEN/' + image_path.split('/')[-1] not in cropped_image_paths:
                image_paths.append(image_path)
                
        if image_paths is None:
            continue

        # split image numbers
        total_base = 0
        if len(image_paths) > base_number:
            total_base = len(image_paths) // base_number
        extra_base = len(image_paths) % base_number
        
        # add processing
        for one_total in range(total_base):
            p = Process(target=process_one, args=(index, bfm, kpt_ind, triangles, uv_coords, head_path, save_paths[index], 
                                        image_paths[one_total*base_number:(one_total+1)*base_number]))
            base_process.append(p)
        p = Process(target=process_one, args=(index, bfm, kpt_ind, triangles, uv_coords, head_path, save_paths[index], 
                                        image_paths[total_base*base_number:total_base*base_number+extra_base]))
        base_process.append(p)
        
    # start processing
    for p in base_process:
        p.start()
    for p in base_process:
        p.join()
    