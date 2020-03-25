#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def calc_percentiles(ln_total_age,ln_ms_age,initial_mass,final_mass,
                     ln_cooling_age,high_perc,low_perc):
    
    total_age_median = np.nanpercentile((10**ln_total_age)/1e9,50)
    total_age_err_low = total_age_median - np.nanpercentile((10**ln_total_age)/1e9,low_perc)
    total_age_err_high = np.nanpercentile((10**ln_total_age)/1e9,high_perc) - total_age_median
    
    cooling_age_median = np.nanpercentile((10**ln_cooling_age)/1e9,50)
    cooling_age_err_low = total_age_median - np.nanpercentile((10**ln_cooling_age)/1e9,low_perc)
    cooling_age_err_high = np.nanpercentile((10**ln_cooling_age)/1e9,high_perc) - cooling_age_median

