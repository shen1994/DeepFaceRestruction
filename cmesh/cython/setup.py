# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:50:46 2018

@author: shen1994
"""

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
      name='mesh_core_cython',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('mesh_core_cython',
                   sources=['mesh_core_cython.pyx', 'mesh_core.cpp'],
                   language='c++',
                   include_dirs=[np.get_include()])],
     )