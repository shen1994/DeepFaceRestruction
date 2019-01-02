# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:10:11 2018

@author: shen1994
"""

import os
import cv2
import cmesh
import face_detect
import numpy as np
import tensorflow as tf
from color_correction import color_hist_match
from color_correction import adain

# from utils.render import render_texture

def process_bbox(bboxes, image_shape, prop=0.16):
        
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i, 0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        d_w, d_h = int(w * prop), int(h * prop)
        y0 = y0 - d_w; y1 = y1 + d_w
        x0 = x0 - d_h; x1 = x1 + d_h
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
    
def generate_texcoord(uv_coords_path, tex_h, tex_w):

    uv_coords = np.loadtxt(uv_coords_path)
    texcoord = np.zeros_like(uv_coords) 
    texcoord[:,0] = uv_coords[:,0]*(tex_h - 1)
    texcoord[:,1] = uv_coords[:,1]*(tex_w - 1)
    texcoord[:,1] = tex_w - texcoord[:,1] - 1
    texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1))))
            
    return texcoord
    
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
    
def get_colors_from_texture(texture, face_ind, resolution_op):

    all_colors = np.reshape(texture, [resolution_op**2, -1])
    colors = all_colors[face_ind, :]

    return colors
    
def run_ref_image(image_path, uv_kpt_ind, face_ind, triangles,
                  pnet, rnet, onet, x, y, Tsess,
                  minsize=30, threshold=[0.6, 0.7, 0.7], factor=0.709, best_score=0.7, 
                  uv_h=256, uv_w=256, image_h=256, image_w=256):
    
    input_image = cv2.imread(image_path, 1)
    boxes, pnts = face_detect.detect_face(input_image, minsize, 
                                          pnet, rnet, onet, threshold, factor)    
    faces = process_bbox(boxes, input_image.shape)
    
    for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
        
        if conf_score > best_score:
        
            det_face = input_image[int(x0):int(x1), int(y0):int(y1), :]
            det_face = cv2.resize(det_face, (256,256)) / 255.
                     
            ref_pos = Tsess.run(y, feed_dict={x: det_face[np.newaxis, :,:,:]})
            ref_pos = np.squeeze(ref_pos)
            max_pos = image_h
            ref_pos = ref_pos * max_pos
            
            ref_texture = cv2.remap(det_face, ref_pos[:,:,:2].astype(np.float32), 
                                    None, interpolation=cv2.INTER_NEAREST, 
                                    borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
            
            break
          
    return ref_texture
    
def to_roi(image, prop):
    
    h, w, _ = image.shape
    p_h = h * prop
    p_w = w * prop
    d_h = int((h - p_h) / 2.)
    d_w = int((w - p_w) / 2.)

    return image[d_h:h-d_h, d_w:w-d_w], d_h, d_w

def image_filter(image, template_image, prop=0.98, kernel_size=5, theta=1.5, color_mode=0, is_origin=False):
    
    '''
    prop < 1.
    '''
    
    ver_image, d_h, d_w = to_roi(image, prop)
    
    t_h, t_w = template_image.shape[0], template_image.shape[1]
    
    if color_mode == 0:
        pass
    elif color_mode == 1:
        ver_image = color_hist_match(ver_image, template_image[d_h:t_h-d_h, d_w:t_w-d_w])
    else:
        ver_image = adain(ver_image, template_image[d_h:t_h-d_h, d_w:t_w-d_w])
        
    template_image[d_h:t_h-d_h, d_w:t_w-d_w] = ver_image

    if is_origin:
        return template_image
      
    half_size = kernel_size // 2
    if t_h-d_h*4-half_size*2 < 0 or t_w-d_w*4-half_size*2 < 0:
        return cv2.GaussianBlur(template_image, (kernel_size, kernel_size), theta)
    
    s_mask = np.zeros_like(template_image, dtype=np.uint8)
    ss_mask = np.zeros((t_h, t_w), dtype=np.uint8)
    s_mask[d_h*2+half_size:t_h-d_h*2-half_size, d_w*2+half_size:t_w-d_w*2-half_size] = \
        template_image[d_h*2+half_size:t_h-d_h*2-half_size, d_w*2+half_size:t_w-d_w*2-half_size]
    ss_mask[d_h*2:t_h-d_h*2, d_w*2:t_w-d_w*2] = np.full((t_h-d_h*4, t_w-d_w*4), 255, dtype=np.uint8)
    t_mask = template_image - s_mask
    t_mask = cv2.GaussianBlur(t_mask, (kernel_size, kernel_size), theta)
            
    ss_mask = cv2.bitwise_not(ss_mask)
    conbine_image = cv2.bitwise_and(t_mask, t_mask, mask=ss_mask)
    conbine_image[d_h*2:t_h-d_h*2, d_w*2:t_w-d_w*2] = template_image[d_h*2:t_h-d_h*2, d_w*2:t_w-d_w*2]

    return conbine_image

def run_one_image(input_image, uv_kpt_ind, face_ind, triangles, uv_coords, 
                  pnet, rnet, onet, x, y, Tsess, ref_texture, uv_whole_face, blend_factor=0.35,
                  minsize=30, threshold=[0.6, 0.7, 0.7], factor=0.709, best_score=0.7, 
                  uv_h=256, uv_w=256, image_h=256, image_w=256):
    
    output_image = input_image.copy()
    boxes, pnts = face_detect.detect_face(input_image, minsize, 
                                          pnet, rnet, onet, threshold, factor)    
    faces = process_bbox(boxes, input_image.shape)
    is_face = False
    for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
        
        if conf_score > best_score:
            
            is_face = True
        
            det_face = input_image[int(x0):int(x1), int(y0):int(y1), :]
            template_face = det_face.copy()          
            face_shape = (int(y1)-int(y0), int(x1)-int(x0))
            det_face = cv2.resize(det_face, (256,256)) / 255.
                     
            pos = Tsess.run(y, feed_dict={x: det_face[np.newaxis, :,:,:]})
            pos = np.squeeze(pos)
            max_pos = image_h * 1.1
            pos = pos * max_pos
            
            vertices = get_vertices(pos, face_ind, uv_h)
            
            vis_colors = np.ones((vertices.shape[0], 1))
            face_mask = cmesh.render.render_texture(vertices.T, vis_colors.T, triangles.T, image_h, image_w, c = 1)
            face_mask = np.squeeze(face_mask > 0).astype(np.float32)
            '''
            texture = cv2.remap(det_face, pos[:,:,:2].astype(np.float32), 
                        None, interpolation=cv2.INTER_NEAREST, 
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

            new_texture = texture*(1. - uv_whole_face[:,:,np.newaxis]) + ref_texture*uv_whole_face[:,:,np.newaxis]
            '''
            new_texture = ref_texture
            new_colors = get_colors_from_texture(new_texture, face_ind, uv_h)
            new_image = cmesh.render.render_texture(vertices.T, new_colors.T, triangles.T, image_h, image_w, c = 3)
            
            
            new_image = blend_factor * det_face*face_mask[:,:,np.newaxis] + (1-blend_factor) * new_image*face_mask[:,:,np.newaxis]
            # new_image = color_hist_match(new_image, det_face*face_mask[:,:,np.newaxis])
            
            new_image = det_face*(1.- face_mask[:,:,np.newaxis]) + new_image

            vis_ind = np.argwhere(face_mask>0)
            vis_min = np.min(vis_ind, 0)
            vis_max = np.max(vis_ind, 0)
            center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
        
            output = cv2.seamlessClone((new_image*255.).astype(np.uint8), (det_face*255.).astype(np.uint8), 
                                       (face_mask*255.).astype(np.uint8), center, cv2.NORMAL_CLONE)
            
            temp_ver_image = cv2.resize(output, face_shape)
            last_ver_image = image_filter(temp_ver_image, template_face)
            output_image[int(x0):int(x1), int(y0):int(y1)] = last_ver_image
          
    return is_face, output_image

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # load params
    uv_kpt_ind = np.loadtxt("images/uv_kpt_ind.txt").astype(np.int32)
    face_ind = np.loadtxt("images/face_ind.txt").astype(np.int32)
    triangles = np.loadtxt("images/triangles.txt").astype(np.int32)
    uv_coords = generate_texcoord("images/uv_coords.txt", 256, 256)
    uv_whole_face = cv2.imread('images/uv_face_mask.png', 0) / 255.
    #uv_face_eye = imread('images/uv_face_eyes.png', as_grey=True) / 255. 
    #uv_face = imread('images/uv_face.png', as_grey=True) / 255.
    #uv_whole_face = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)
    
    # load model
    pnet, rnet, onet = load_detect_model()
    tx, ty, tsess = load_3dface_model()
    
    # load ref image
    ref_texture = run_ref_image("ref.jpg", uv_kpt_ind, face_ind, triangles, 
                      pnet, rnet, onet, tx, ty, tsess)

    print('load model ok!')
    '''
    image_path = 'images/tfd.jpg'
    is_face, ver_image = run_one_image(cv2.imread(image_path, 1), uv_kpt_ind, face_ind, triangles, uv_coords, 
                                  pnet, rnet, onet, tx, ty, tsess, ref_texture, uv_whole_face)
    cv2.imwrite(os.path.join("images", "test" + '_mask.jpg'), ver_image)
    '''
    # camera settins
    vedio_shape = [1920, 1080]
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vedio_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vedio_shape[1])
    cv2.namedWindow("Deep3DFace", cv2.WINDOW_NORMAL)

    while(True):
        
        _, o_image = cap.read()

        is_face, ver_image = run_one_image(o_image, uv_kpt_ind, face_ind, triangles, uv_coords, 
                                  pnet, rnet, onet, tx, ty, tsess, ref_texture, uv_whole_face, blend_factor=0.35)

        cv2.imshow("Deep3DFace", np.hstack((o_image, ver_image)))
            
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        
    cv2.destroyAllWindows()

    '''
    v_cap = cv2.VideoCapture("data_A.mp4")
    
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(v_cap.get(cv2.CAP_PROP_FOURCC))
    size = (int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter("data_B.mp4", fourcc, fps, size, True)

    counter = 0    
    success, frame = v_cap.read()
    while success :
        try:
            is_face, ver_image = run_one_image(frame, uv_kpt_ind, face_ind, triangles, uv_coords, 
                                  pnet, rnet, onet, tx, ty, tsess, ref_texture, uv_whole_face)
            if is_face:
                cv2.imwrite('swap/'+str(counter)+'.jpg', ver_image)
                videoWriter.write(ver_image)
            else:
                videoWriter.write(frame)
        except Exception:
            pass
        is_face, ver_image = run_one_image(frame, uv_kpt_ind, face_ind, triangles, uv_coords, 
                                  pnet, rnet, onet, tx, ty, tsess, ref_texture, uv_whole_face)
        if is_face:
            cv2.imwrite('swap/'+str(counter)+'.jpg', ver_image)
            videoWriter.write(ver_image)
        else:
            videoWriter.write(frame)

        counter += 1
        print('PROCESSING %d OK' %(counter))
        cv2.waitKey(1000//int(fps))
        success, frame = v_cap.read()
        
    print('PROCESSING IS OK!')
    v_cap.release()
    videoWriter.release()
    '''
    