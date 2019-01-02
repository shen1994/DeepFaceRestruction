# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:52:19 2018

@author: shen1994
"""

import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from tmodel import PRNet

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    x, pos = PRNet()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint("model")
        saver.restore(sess, ckpt)
        
        print(x.name, pos.name)
        '''
        nodes = [node.name for node in sess.graph.as_graph_def().node]
        node_name = 'PRNet/Conv2d_transpose_16'
        for node in nodes:
            if node[:len(node_name)] == node_name:
                print(node)
        '''
        tf.train.write_graph(sess.graph.as_graph_def(), 'model', 'model_graph.pb')
        freeze_graph.freeze_graph('model/model_graph.pb',
                                  '', 
                                  False, 
                                  ckpt, 
                                  pos.name[:-2], 
                                  'save/restore_all', 
                                  'save/Const:0', 
                                  'model/pico_3dFace_model.pb', 
                                  False, 
                                  "")
