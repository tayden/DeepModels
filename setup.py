#!/usr/bin/env python

from setuptools import setup

setup(name='Deep-Models',
      version='0.0.1',
      description='Keras deep learning model implementations',
      url='http://github.com/tayden/DeepModels',
      author='Taylor Denouden',
      author_email='taylordenouden@gmail.com',
      license='MIT',
      packages=['deep_models'],
      install_requires=[
            'keras',
      ])