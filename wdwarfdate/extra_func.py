#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def calc_percentiles(ln_ms_age, ln_cooling_age, ln_total_age, initial_mass, 
                     final_mass, high_perc, low_perc):
    
    ms_age_median = np.nanpercentile((10**ln_ms_age)/1e9,50)
    ms_age_err_low = ms_age_median - np.nanpercentile((10**ln_ms_age)/1e9,low_perc)
    ms_age_err_high = np.nanpercentile((10**ln_ms_age)/1e9,high_perc) - ms_age_median
    
    cooling_age_median = np.nanpercentile((10**ln_cooling_age)/1e9,50)
    cooling_age_err_low = cooling_age_median - np.nanpercentile((10**ln_cooling_age)/1e9,low_perc)
    cooling_age_err_high = np.nanpercentile((10**ln_cooling_age)/1e9,high_perc) - cooling_age_median
    
    total_age_median = np.nanpercentile((10**ln_total_age)/1e9,50)
    total_age_err_low = total_age_median - np.nanpercentile((10**ln_total_age)/1e9,low_perc)
    total_age_err_high = np.nanpercentile((10**ln_total_age)/1e9,high_perc) - total_age_median
    
    initial_mass_median = np.nanpercentile(initial_mass,50)
    initial_mass_err_low = initial_mass_median - np.nanpercentile(initial_mass,low_perc)
    initial_mass_err_high = np.nanpercentile(initial_mass,high_perc) - initial_mass_median
    
    final_mass_median = np.nanpercentile(final_mass,50)
    final_mass_low = final_mass_median - np.nanpercentile(final_mass,low_perc)
    final_mass_high = np.nanpercentile(final_mass,high_perc) - final_mass_median

    return [ms_age_median,ms_age_err_low,ms_age_err_high,
            cooling_age_median,cooling_age_err_low,cooling_age_err_high,
            total_age_median,total_age_err_low,total_age_err_high,
            initial_mass_median,initial_mass_err_low,initial_mass_err_high,
            final_mass_median,final_mass_low,final_mass_high]
