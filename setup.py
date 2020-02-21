#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='trec2019',
      version='0.0.1',
      description='TREC 2019 Deep Learning',
      author='Kyoung-Rok Jang',
      author_email='kyoungrok.jang@gmail.com',
      url='https://github.com/kyoungrok0517/trec-2019-deep-learning',
      install_requires=[
            'pytorch-lightning',
            'torch',
            'transformers',
            'h5py',
            'tqdm',
            'pyarrow',
            'python-snappy'
      ],
      packages=find_packages()
      )

