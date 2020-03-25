#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:48:04 2020

@author: rociokiman
"""
import numpy as np
from wdwarfdate.ms_age import calc_ms_age

def test_calc_ms_age():
    N = 10
    sigma = 0.5
    initial_mass_dist = np.array([np.random.normal(np.random.rand()*5+1,sigma,1000) for i in range(N)])
    ms_age_dist = calc_ms_age(initial_mass_dist,feh='p0.00',vvcrit='0.0')
    
    ms_age_median = np.array([np.nanpercentile(ms_age,50) for ms_age in ms_age_dist])
    ms_age_err_high = np.array([np.nanpercentile(ms_age,84) for ms_age in ms_age_dist]) - ms_age_median
    
    
    assert(all(ms_age_median/1e9 < 13.8))
    assert(all(ms_age_median/1e9+ms_age_err_high/1e9 < 13.8))
    
