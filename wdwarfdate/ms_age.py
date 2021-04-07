import numpy as np
from astropy.table import Table
from scipy import interpolate
import os
import inspect

def get_isochrone_model(feh,vvcrit):
    '''
    Interpolates MIST isochrones to get a function that gives initial mass
    as a function of main sequence age. This function is used in the 'bayesian'
    method.
    
    Parameters
    ----------
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00' 
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'
    
    Returns
    -------
    f_initial_mass : interpolated function. Calculates initial mass from
                     a main sequence age.
    model_initial_mass : array. List of initial masses used to create the 
                         f_initial_mass model.
    model_ms_age : array. List of main sequence ages used to create the 
                   f_initial_mass model.
    '''
    #Load isochrone
    file_path = 'Models/MIST/MIST_v1.2_feh_'
    path = file_path + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS_sum.csv'
    path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
    filepath = os.path.join(path1,path)
        
    table_model = Table.read(filepath)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = np.log10(table_model['ms_age'])
    
    #Interpolate model from isochrone
    f_initial_mass = interpolate.interp1d(model_ms_age,model_initial_mass,
                                          fill_value=np.nan)
    
    return f_initial_mass,model_initial_mass,model_ms_age

def calc_ms_age(initial_mass_dist,feh,vvcrit):
    '''
    Calculates a main sequence age distribution of the white dwarf's progenitor 
    for each white dwarf from the initial mass distribution using 
    MIST isochrones. This function is used in the 'freq' method.
    
    Parameters
    ----------
    initial_mass_dist : list of arrays. List of initial masses distributions 
                        for each white dwarf progenitor.
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00' 
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'
    
    Returns
    -------
    ms_age_dist : list of arrays. List of main sequence age distributions 
                       for each white dwarf progenitor.
    '''

    #Load isochrone model
    file_path = 'Models/MIST/MIST_v1.2_feh_'
    path = file_path + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS_sum.csv'
    path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
    filepath = os.path.join(path1,path)

    table_model = Table.read(filepath)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = table_model['ms_age']

    #Interpolate model using isochrone values
    f_ms_age = interpolate.interp1d(model_initial_mass, 
                                    model_ms_age, kind='cubic')
    
    #Replace with nan all the values of the initial_mass not included in 
    #the interpolation model
    initial_mass_dist_copy = np.copy(initial_mass_dist)
    mask_nan = np.isnan(initial_mass_dist_copy)
    initial_mass_dist_copy[mask_nan] = 2
    mask = np.logical_or(np.min(model_initial_mass) > initial_mass_dist_copy,
                         np.max(model_initial_mass) < initial_mass_dist_copy)
    initial_mass_dist_copy[mask] = np.nan
    initial_mass_dist_copy[mask_nan] = np.nan
    
    #Use the interpolated model to calculate main sequence age
    ms_age_dist = []
    for initial_mass_dist_i in initial_mass_dist_copy:
        ms_age_dist_i = np.array([f_ms_age(x) for x in initial_mass_dist_i])
        ms_age_dist.append(ms_age_dist_i)
    
    ms_age_dist = np.array(ms_age_dist)
    
    #Replace with nan all the values of ms_age bigger than the age of the 
    #Universe
    mask_nan = np.isnan(ms_age_dist)
    ms_age_dist[mask_nan] = -1
    
    #mask for the prior age less than the current age
    #of hte universe
    #mask = ms_age_dist/1e9 > 1e6#13.8
    #ms_age_dist[mask] = np.nan
    
    ms_age_dist[mask_nan] = np.nan
    
    return np.array(ms_age_dist)
