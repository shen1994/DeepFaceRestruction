# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:10:07 2018

@author: shen1994
"""

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

def res_block(x, num_outputs, kernel_size=4, stride=1):
    assert num_outputs % 2 == 0
    shortcut = x
    if stride != 1 or int(x.get_shape()[3]) != num_outputs:
        shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                              activation_fn=None, normalizer_fn=None)
    x = tcl.conv2d(x, num_outputs/2, kernel_size=1, stride=1, padding='SAME')
    x = tcl.conv2d(x, num_outputs/2, kernel_size=kernel_size, stride=stride, padding='SAME')
    x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, padding='SAME', 
                   activation_fn=None, normalizer_fn=None)
    x += shortcut
    x = tcl.batch_norm(x)
    x = tf.nn.relu(x)
    
    return x
            
def PRNet(image_shape=(256, 256, 3), output_shape=(256, 256,3), size=16, is_training=True):
    
    with tf.name_scope('3dface') as scope:
        x = tf.placeholder(shape=(None, image_shape[0], image_shape[1], image_shape[2]), 
                           dtype=tf.float32, name='x')
        m = tf.placeholder(shape=(None, image_shape[0], image_shape[1]), 
                              dtype=tf.float32, name='m')
        y = tf.placeholder(shape=(None, output_shape[0], output_shape[1], output_shape[2]), 
                           dtype=tf.float32, name='y')
    with tf.name_scope('PRNet') as scope: 
        with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
            with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                           normalizer_fn=tcl.batch_norm, biases_initializer=None, 
                           padding='SAME', weights_regularizer=tcl.l2_regularizer(0.0002)):
                # encoder
                se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) # 256 x 256 x 16
                se = res_block(se, num_outputs=size * 2, kernel_size=4, stride=2) # 128 x 128 x 32
                se = res_block(se, num_outputs=size * 2, kernel_size=4, stride=1) # 128 x 128 x 32
                se = res_block(se, num_outputs=size * 4, kernel_size=4, stride=2) # 64 x 64 x 64
                se = res_block(se, num_outputs=size * 4, kernel_size=4, stride=1) # 64 x 64 x 64
                se = res_block(se, num_outputs=size * 8, kernel_size=4, stride=2) # 32 x 32 x 128
                se = res_block(se, num_outputs=size * 8, kernel_size=4, stride=1) # 32 x 32 x 128
                se = res_block(se, num_outputs=size * 16, kernel_size=4, stride=2) # 16 x 16 x 256
                se = res_block(se, num_outputs=size * 16, kernel_size=4, stride=1) # 16 x 16 x 256
                se = res_block(se, num_outputs=size * 32, kernel_size=4, stride=2) # 8 x 8 x 512
                se = res_block(se, num_outputs=size * 32, kernel_size=4, stride=1) # 8 x 8 x 512
                # decoder
                pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1) # 8 x 8 x 512 
                pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2) # 16 x 16 x 256 
                pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256 
                pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256 
                pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2) # 32 x 32 x 128 
                pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128 
                pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128 
                pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2) # 64 x 64 x 64 
                pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64 
                pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64 
                    
                pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2) # 128 x 128 x 32
                pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1) # 128 x 128 x 32
                pd = tcl.conv2d_transpose(pd, size, 4, stride=2) # 256 x 256 x 16
                pd = tcl.conv2d_transpose(pd, size, 4, stride=1) # 256 x 256 x 16

                pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                pos = tcl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn = tf.nn.sigmoid)
                
    with tf.name_scope('loss') as scope:        
        locc = tf.square(y - pos)
        loss_x = tf.reduce_mean(tf.multiply(locc[:,:,:,0], m))
        loss_y = tf.reduce_mean(tf.multiply(locc[:,:,:,1], m))
        loss_z = tf.reduce_mean(tf.multiply(locc[:,:,:,2], m))
        loss = loss_x + loss_y + loss_z
        
    with tf.name_scope('optimizer') as scope:
        o_learning_rate = 1e-4
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(o_learning_rate, global_step, 40000, 0.1, staircase=True)		
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
    return x, m, y, pos, loss, optimizer            
            