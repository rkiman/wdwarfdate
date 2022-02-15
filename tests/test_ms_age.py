#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:48:04 2020

@author: rociokiman
"""
import numpy as np
from wdwarfdate.ms_age import calc_ms_age


def test_calc_ms_age():
    n = 1000
    sigma = 0.5
    initial_mass = np.random.normal(np.random.rand()*5+1,sigma, n)

    ms_age = calc_ms_age(initial_mass, feh='p0.00', vvcrit='0.0')

    assert(all(ms_age/1e9 < 15))
