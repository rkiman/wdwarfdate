#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.extra_func import calc_percentiles
import numpy as np

def test_calc_percentiles():
    
    dist = np.random.normal(10,5,1000)
    log_dist = np.log10(dist)
    high_perc, low_perc = 84,16
    results = calc_percentiles(log_dist, log_dist, log_dist, dist, 
                               dist, high_perc, low_perc, datatype='yr')
    assert np.isclose(results[0],10,atol=1e-1)
    assert np.isclose(results[3],10,atol=1e-1)
    assert np.isclose(results[6],10,atol=1e-1)
    



