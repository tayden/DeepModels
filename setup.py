#!/usr/bin/env python

from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

with open('VERSION') as file:
    version = file.read()

setup(name='Deep-Models',
      version=version,
      description='Keras deep learning model implementations',
      long_description=long_description,
      url='http://github.com/tayden/DeepModels',
      author='Taylor Denouden',
      author_email='taylordenouden@gmail.com',
      license='MIT',
      packages=['deep_models'],
      install_requires=[
            'keras',
      ])
