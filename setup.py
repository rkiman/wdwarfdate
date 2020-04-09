#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

with open("wdwarfdate/version.py", "r") as f:
    exec(f.read())

setup(name='wdwarfdate_code',
      version=__version__,
      description='Calculates white dwarfs ages from Gaia photometry.',
      url='https://github.com/rkiman/wdwarfdate',
      author='Rocio Kiman',
      author_email='rociokiman@gmail.com',
      #license='MIT',
      packages=['wdwarfdate'],
      install_requires=['numpy','astropy','matplotlib','emcee','corner','scipy'],
      zip_safe=False)

