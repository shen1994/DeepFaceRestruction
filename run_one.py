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

from skimage.io import imsave
from skimage.transform import resize
from utils.render_app import get_visibility
from utils.render_app import get_uv_mask
from utils.render_app import get_depth_image
from utils.write import write_obj_with_texture
from utils.write import write_obj_with_colors
from utils.estimate_pose import estimate_pose
from utils.cv_plot import plot_kpt
from utils.cv_plot import plot_vertices
from utils.cv_plot import plot_pose_box

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

def run_one_image(bfm, uv_coords, uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  image_path, mat_path, save_folder, name, 
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
    
    # 5. deal uv map
    # run model to get uv_map
    pos = uv_position_map / image_h
    max_pos = image_h
    pos = pos * max_pos

    # 6. get useful vertices
    vertices = get_vertices(pos, face_ind, uv_h)
    save_vertices = vertices.copy()
    save_vertices[:,1] = image_h - 1 - save_vertices[:,1]

    # 7. get colors
    colors = get_colors(cropped_image, vertices)
    write_obj_with_colors(os.path.join(save_folder, name + '_c.obj'), 
                          save_vertices, triangles, colors)
    
    print('color 3d face is ok!')

    # 8. get texture
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

    # 9. get depth image
    depth_image = get_depth_image(vertices, triangles, image_h, image_w, True)
    imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
    
    print('depth image is ok!') 
    
    # get restore size
    restore_top = int(center[0]-size/2)
    restore_bottom = int(center[0]+size/2)
    restore_left = int(center[1]-size/2)
    restore_right = int(center[1]+size/2)
    restore_w = restore_right - restore_left
    restore_h = restore_bottom - restore_top
    
    # 10. get landmarks
    t_image = (cropped_image*255.).astype(np.uint8)
    kpt = get_landmarks(pos, uv_kpt_ind)
    kpt_origin = plot_kpt(cropped_image, kpt).astype(np.uint8)
    kpt_gray = cv2.cvtColor(kpt_origin, cv2.COLOR_RGB2GRAY) 
    ret, kpt_mask = cv2.threshold(kpt_gray, 127, 255, cv2.THRESH_BINARY) 
    kpt_mask = cv2.bitwise_not(kpt_mask)
    kpt_and = cv2.bitwise_and(t_image, t_image, mask=kpt_mask)
    kpt_image = cv2.add(kpt_and, kpt_origin)
    imsave(os.path.join(save_folder, name + '_kpt.jpg'), kpt_image/255.)
    
    print('kpt image is ok!')

    # 10.0 restore kpt image  
    resize_kpt_image = resize(kpt_image, (restore_w, restore_h))
    rt_kpt_image = image.copy()
    rt_kpt_image[restore_left:restore_right, restore_top:restore_bottom] = resize_kpt_image
    imsave(os.path.join(save_folder, name + '_r_kpt.jpg'), rt_kpt_image)
    
    print('kpt fll image is ok!')
       
    # 11. get mask
    t_image = (cropped_image*255.).astype(np.uint8)
    ver_origin = plot_vertices(cropped_image, vertices).astype(np.uint8)
    ver_gray = cv2.cvtColor(ver_origin, cv2.COLOR_RGB2GRAY) 
    ret, ver_mask = cv2.threshold(ver_gray, 127, 255, cv2.THRESH_BINARY) 
    ver_mask = cv2.bitwise_not(ver_mask)
    ver_and = cv2.bitwise_and(t_image, t_image, mask=ver_mask)
    ver_image = cv2.add(ver_and, ver_origin)
    imsave(os.path.join(save_folder, name + '_ver.jpg'), ver_image/255.) 
     
    print('vertices image is ok!')
    
    # 11.0 restore ver image
    resize_ver_image = resize(ver_image, (restore_w, restore_h))
    rt_ver_image = image.copy()
    rt_ver_image[restore_left:restore_right, restore_top:restore_bottom] = resize_ver_image
    imsave(os.path.join(save_folder, name + '_r_ver.jpg'), rt_ver_image)
    
    print('vertices full image is ok!')

    # 12. get camera map
    camera_matrix, pose = estimate_pose(vertices)
    imsave(os.path.join(save_folder, name + '_cam.jpg'), plot_pose_box(cropped_image, camera_matrix, kpt)/255.)
    
    print('camera image is ok!')


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
    
    image_path = 'images/300W_LP/AFW/AFW_134212_1_1.jpg'
    mat_path = 'images/300W_LP/AFW/AFW_134212_1_1.mat'
   
    run_one_image(bfm, uv_coords, uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  image_path, mat_path, 'images', 'test')
    