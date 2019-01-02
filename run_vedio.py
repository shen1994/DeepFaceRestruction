# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:10:11 2018

@author: shen1994
"""

import os
import cv2
import glob
import face_detect
import numpy as np
import tensorflow as tf

from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from utils.cv_plot import plot_kpt
from utils.cv_plot import plot_vertices

def process_bbox(bboxes, image_shape):
        
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i, 0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        length = (w + h) / 2
        center = (int((x1+x0)/2), int((y1+y0)/2))
        new_x0 = np.max([0, center[0]-length//2])
        new_x1 = np.min([image_shape[0], center[0]+length//2])
        new_y0 = np.max([0, center[1]-length//2])
        new_y1 = np.min([image_shape[1], center[1]+length//2])
        bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
    
    return bboxes
    
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
    
def load_detect_model():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = face_detect.create_mtcnn(sess, None)
    return pnet, rnet, onet
    
def load_3dface_model():
    
    tface_graph_def = tf.GraphDef()
    tface_graph_def.ParseFromString(open("model/pico_3dFace_model.pb", "rb").read())
    tf.import_graph_def(tface_graph_def, name="")
    tface_sess = tf.Session()
    tface_sess.graph.get_operations()
    tx = tface_sess.graph.get_tensor_by_name("3dface/x:0")
    ty = tface_sess.graph.get_tensor_by_name("PRNet/Conv2d_transpose_16/Sigmoid:0")
    
    return tx, ty, tface_sess 

def run_one_image(image_path, uv_kpt_ind, face_ind, triangles, s_uv_coords,
                  pnet, rnet, onet, x, y, Tsess,
                  minsize=30, threshold=[0.6, 0.7, 0.7], factor=0.709, best_score=0.7, 
                  uv_h=256, uv_w=256, image_h=256, image_w=256):
    
    input_image = cv2.imread(image_path, 1)
    output_image = input_image.copy()
    boxes, pnts = face_detect.detect_face(input_image, minsize, 
                                          pnet, rnet, onet, threshold, factor)    
    faces = process_bbox(boxes, input_image.shape)
    
    for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
        
        if conf_score > best_score:
        
            det_face = input_image[int(x0):int(x1), int(y0):int(y1), :]
                      
            face_shape = (int(y1)-int(y0), int(x1)-int(x0))
            det_face = cv2.resize(det_face, (256,256)) / 255.
                     
            pos = Tsess.run(y, feed_dict={x: det_face[np.newaxis, :,:,:]})
            pos = np.squeeze(pos)
            max_pos = image_h
            pos = pos * max_pos
                
            vertices = get_vertices(pos, face_ind, uv_h)
            
            from utils.write import write_obj_with_colors
            save_vertices = vertices.copy()
            save_vertices[:,1] = image_h - 1 - save_vertices[:,1]
            colors = get_colors(det_face, vertices)
            write_obj_with_colors(os.path.join('images', 'test' + '_c.obj'), 
                          save_vertices, triangles, colors)
            
            t_image = (det_face*255.).astype(np.uint8)
            kpt = get_landmarks(pos, uv_kpt_ind)
            kpt_origin = plot_kpt(det_face, kpt).astype(np.uint8)
            kpt_gray = cv2.cvtColor(kpt_origin, cv2.COLOR_RGB2GRAY) 
            ret, kpt_mask = cv2.threshold(kpt_gray, 127, 255, cv2.THRESH_BINARY) 
            kpt_mask = cv2.bitwise_not(kpt_mask)
            kpt_and = cv2.bitwise_and(t_image, t_image, mask=kpt_mask)
            kpt_image = cv2.add(kpt_and, kpt_origin)
            imsave(os.path.join('images', 'test' + '_kpt.jpg'), kpt_image/255.)
                                       
            t_image = (det_face*255.).astype(np.uint8)
            ver_origin = plot_vertices(det_face, vertices).astype(np.uint8)
            ver_gray = cv2.cvtColor(ver_origin, cv2.COLOR_RGB2GRAY) 
            ret, ver_mask = cv2.threshold(ver_gray, 127, 255, cv2.THRESH_BINARY) 
            ver_mask = cv2.bitwise_not(ver_mask)
            ver_and = cv2.bitwise_and(t_image, t_image, mask=ver_mask)
            ver_image = cv2.add(ver_and, ver_origin)
            imsave(os.path.join('images', 'test' + '_ver.jpg'), ver_image/255.)
        
            resize_ver_image = cv2.resize(ver_image, face_shape)
            
            output_image[int(x0):int(x1), int(y0):int(y1)] = resize_ver_image
          
    return output_image / 255.

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # load params
    uv_kpt_ind = np.loadtxt('images/uv_kpt_ind.txt').astype(np.int32)
    face_ind = np.loadtxt("images/face_ind.txt").astype(np.int32)
    triangles = np.loadtxt("images/triangles.txt").astype(np.int32)
    s_uv_coords = generate_uv_coords(face_ind, 256)
    
    # load model
    pnet, rnet, onet = load_detect_model()
    tx, ty, tsess = load_3dface_model()

    print('load model ok!')
    ver_image = run_one_image('images/test.jpg', uv_kpt_ind, face_ind, triangles, s_uv_coords, 
                      pnet, rnet, onet, tx, ty, tsess)
    '''
    image_paths = glob.glob(os.path.join("TrainingData", '*.jpg'))
    counter = 0
    for image_path in image_paths:
    # image_path = 'images/300W_LP/AFW_GEN/AFW_111076519_1_1.jpg'
        counter += 1
        ver_image = run_one_image(image_path, uv_kpt_ind, face_ind, triangles, s_uv_coords, 
                      pnet, rnet, onet, tx, ty, tsess)
        
        imsave(os.path.join("TrainingData", "test" + '_mask_' + str(counter) + '.jpg'), ver_image)
    '''
    