#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from distutils.sysconfig import *
from distutils.util import *
from Cython.Distutils import build_ext
import numpy

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

py_inc = [get_python_inc()]
np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

setup(
    name='cdtw',
    version='0.1.1',
    description='Fast, inflexible dtw in python/cython.',
    long_description=readme + '\n\n' + history,
    author='Maarten Versteegh',
    author_email='maartenversteegh@gmail.com',
    url='https://github.com/mwv/cdtw',
    install_requires=['numpy'],
    license="GPLv3",
    zip_safe=False,
    keywords='cdtw',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('cdtw', ['src/cdtw.pyx'],
                           include_dirs=py_inc + np_inc,)],
                           # extra_compile_args=["-O3 -shared -pthread -fPIC"
                           #                     "-fwrapv -Wall "
                           #                     "-fno-strict-aliasing"])],
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')]
)
