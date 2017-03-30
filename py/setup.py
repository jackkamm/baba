#!/usr/bin/env python
from distutils.core import setup

setup(name='baba',
      version='0.1',
      description="Tensor decomposition of ABBA-BABA statistics",
      author='Jack Kamm',
      author_email='jackkamm@gmail.com',
      packages=['baba'],
      install_requires=['autograd>=1.1.7', 'numpy>=1.9.0', 'scipy', 'cached_property>=1.3'],
      )
