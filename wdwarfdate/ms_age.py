import numpy as np
from astropy.table import Table
from scipy import interpolate

def calc_ms_age(initial_mass_dist,feh,vvcrit):
    ms_age_dist = []
    
    file = 'Models/MIST/MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_EEPS_sum.csv'
    table_model = Table.read(file)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = table_model['ms_age']

    f_ms_age = interpolate.interp1d(model_initial_mass, 
                                    model_ms_age, kind='cubic')
    
    #Replace with nan all the values of the initial_mass not included in 
    #the interpolation model
    initial_mass_dist_copy = np.copy(initial_mass_dist)
    mask = np.logical_or(np.min(model_initial_mass) > initial_mass_dist_copy,
                         np.max(model_initial_mass) < initial_mass_dist_copy)
    initial_mass_dist_copy[mask] = np.nan

    for initial_mass_dist_i in initial_mass_dist_copy:
        ms_age_dist_i = np.array([f_ms_age(x) for x in initial_mass_dist_i])
        ms_age_dist.append(ms_age_dist_i)
    
    ms_age_dist = np.array(ms_age_dist)
    
    #Replace with nan all the values of ms_age bigger than the age of the Universe
    mask_nan = np.isnan(ms_age_dist)
    ms_age_dist[mask_nan] = -1
    
    mask = ms_age_dist/1e9 > 13.8
    ms_age_dist[mask] = np.nan
    
    ms_age_dist[mask_nan] = np.nan
    
    return np.array(ms_age_dist)
