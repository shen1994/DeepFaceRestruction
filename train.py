# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:15:27 2018

@author: shen1994
"""

import os 
import cv2
import numpy as np
import tensorflow as tf
from model import PRNet
from generate import Generator
from predictor import Predictor
from show import show_G

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
     
    # define params
    batch_size = 32
    epocs = 100001
    image_shape = (256, 256, 3)
    
    # define model
    x, m, y, pos, loss, optimizer = PRNet()
    
    # define predictor
    v_predictor = Predictor()
    
    # define generator
    train_paths = ['images/300W_LP/AFW_GEN', 'images/300W_LP/HELEN_GEN',
                   'images/300W_LP/IBUG_GEN', 'images/300W_LP/LFPW_GEN',
                   'images/300W_LP/AFW_Flip_GEN', 'images/300W_LP/HELEN_Flip_GEN',
                   'images/300W_LP/IBUG_Flip_GEN', 'images/300W_LP/LFPW_Flip_GEN']
    valid_path = 'images/AFLW2000_GEN'
    mask_path = 'images/uv_weight_mask2.png'
    
    t_generator = Generator(train_paths=train_paths,
                   valid_path=valid_path,
                   mask_path=mask_path,
                   image_shape=image_shape, 
                   batch_size=batch_size).generate(is_training=True)
    v_generator = Generator(train_paths=train_paths,
                   valid_path=valid_path,
                   mask_path=mask_path,
                   image_shape=image_shape, 
                   batch_size=batch_size).generate(is_training=False)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        # initial variables
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        # restore model
        try:
            ckpt = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, ckpt)
        except Exception:
            print('No existed model to use!')
        
        # train data
        step = 0
        total_coss = 0
        while step < epocs:
            
            x_in, y_in, m_in = t_generator.__next__()
            _ = sess.run(optimizer, feed_dict={x: x_in, y: y_in, m: m_in})
            total_coss += sess.run(loss, feed_dict={x: x_in, y: y_in, m: m_in})
            
            if step % 100 == 0:
                
                # show total loss
                print(str(step) + ": train --->" + "cost:" + str(total_coss))
                print("---------------------------------------->")
                total_coss = 0
                
                # show keypoints
                x_ou = v_generator.__next__()
                x_pre_ou = []
                for i in range(16):
                    x_pre_ou.append(v_predictor.predictor(sess, x, pos, x_ou[i]))
                show_G(x_ou[:16], np.array(x_pre_ou), 16, "3DFace")
             
            # save model
            if step % 1000 == 0 and step != 0:
                saver.save(sess, 'model/model%d.ckpt' % step)
            
            # exit programs
            if cv2.waitKey(1) == ord('q'):
                exit()
                
            step += 1    
       