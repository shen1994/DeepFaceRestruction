# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:04:13 2018

@author: shen1994
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .cython import mesh_core_cython

def render_texture(vertices, colors, triangles, h, w, c=3):
    
    image = np.zeros((h, w, c), dtype=np.float32, order='C')
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2, triangles[1,:]] + vertices[2, triangles[2,:]])/3.
    tri_tex = (colors[:, triangles[0,:]] + colors[:, triangles[1,:]] + colors[:, triangles[2,:]])/3.

    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    tri_depth = tri_depth.astype(np.float32).copy()
    tri_tex = tri_tex.astype(np.float32).copy()

    mesh_core_cython.render_texture_core(
                image, vertices, triangles,
                tri_depth, tri_tex, depth_buffer,
                vertices.shape[1], triangles.shape[1], tri_tex.shape[1],
                h, w, c)

    return image
    