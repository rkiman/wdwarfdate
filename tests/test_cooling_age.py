#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.cooling_age import calc_cooling_age
import numpy as np

def test_calc_cooling_age():
    N = 100
    teff_d = np.array([np.random.normal(10000,100,N)])
    logg_d = np.array([np.random.normal(8,0.1,N)])
    cooling_age_d,final_mass_d = calc_cooling_age(teff_d,logg_d,1,'DA')
    assert len(cooling_age_d[0]) == N
    assert len(final_mass_d[0]) == N

    cooling_age_d2,final_mass_d2 = calc_cooling_age(teff_d,logg_d,1,'DB')
    assert len(cooling_age_d2[0]) == N
    assert len(final_mass_d2[0]) == N