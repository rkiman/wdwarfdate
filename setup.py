#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='wdwarfdate_code',
      version="0.1.0",
      description='Calculates white dwarfs ages from Gaia photometry.',
      url='https://github.com/rkiman/wdwarfdate',
      author='Rocio Kiman',
      author_email='rociokiman@gmail.com',
      #license='MIT',
      packages=['wdwarfdate'],
      install_requires=['numpy','astropy','matplotlib','emcee','corner','scipy'],
      zip_safe=False,
      include_package_data=True)

