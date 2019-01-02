# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:33:02 2018

@author: shen1994
"""

import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

cdef extern from "mesh_core.h":
    
    void _render_texture_core(
        float* image, float* vertices, int* triangles,
        float* tri_depth, float* tri_tex, float* depth_buffer,
        int ver_len, int tri_len, int tex_len, int h, int w, int c)
    
def render_texture_core(np.ndarray[float, ndim=3, mode="c"] image not None,
                        np.ndarray[float, ndim=2, mode="c"] vertices not None,
                        np.ndarray[int, ndim=2, mode="c"] triangles not None,
                        np.ndarray[float, ndim=1, mode="c"] tri_depth not None,
                        np.ndarray[float, ndim=2, mode="c"] tri_tex not None,
                        np.ndarray[float, ndim=2, mode="c"] depth_buffer not None,
                        int ver_len, int tri_len, int tex_len, int h, int w, int c):
    _render_texture_core(<float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices),
                         <int*> np.PyArray_DATA(triangles), <float*> np.PyArray_DATA(tri_depth),
                         <float*> np.PyArray_DATA(tri_tex), <float*> np.PyArray_DATA(depth_buffer),
                         ver_len, tri_len, tex_len, h, w, c)
