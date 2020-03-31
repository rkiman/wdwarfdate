#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table
from scipy import interpolate

min_initial_mass_mist = 0.83 - 0.01
max_initial_mass_mist =  7.20 + 0.3

def ifmr(initial_mass,ifmr_model):
    '''
    Define Initial-Final Mass relation
    '''
    #Initialize variables
    initial_mass = np.asarray(initial_mass)
    final_mass = np.copy(initial_mass)*np.nan
    final_mass = np.asarray(final_mass)
    
    if(ifmr_model == 'Cummings_2018_MIST'):
        #Initial-Final mass relation from 
        #Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        #based on MIST isochrones
        mask1 = (initial_mass < 2.85)#(min_initial_mass_mist <= initial_mass) * (initial_mass < 2.85)
        mask2 = (2.85 <= initial_mass) * (initial_mass < 3.60)
        mask3 = (3.60 <= initial_mass) #* (initial_mass < max_initial_mass_mist)
        final_mass[mask1] = initial_mass[mask1] * 0.08 + 0.489
        final_mass[mask2] = initial_mass[mask2] * 0.187 + 0.184
        final_mass[mask3] = initial_mass[mask3] * 0.107 + 0.471
    elif(ifmr_model == 'Cummings_2018_PARSEC'):
        #Initial-Final mass relation from 
        #Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        #based on PARSEC isochrones
        mask1 = (0.87 <= initial_mass) * (initial_mass < 2.8)
        mask2 = (2.8 <= initial_mass) * (initial_mass < 3.65)
        mask3 = (3.65 <= initial_mass) * (initial_mass < 8.20)
        final_mass[mask1] = initial_mass[mask1] * 0.0873 + 0.476
        final_mass[mask2] = initial_mass[mask2] * 0.181 + 0.210
        final_mass[mask3] = initial_mass[mask3] * 0.0835 + 0.565
    elif(ifmr_model == 'Salaris_2009'):
        #Initial-Final mass relation from 
        #Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009).
        
        mask1 = (1.7 <= initial_mass) * (initial_mass < 4)
        mask2 = (4 <= initial_mass) 
        final_mass[mask1] = initial_mass[mask1] * 0.134 + 0.331
        final_mass[mask2] = initial_mass[mask2] * 0.047 + 0.679   
    return final_mass

def get_cooling_model(model_wd):
    
    if(model_wd == 'DA'):
        table_model = np.loadtxt('Models/cooling_models/Table_DA')

    if(model_wd == 'DB'):
        table_model = np.loadtxt('Models/cooling_models/Table_DB')
    
    model_T = table_model[:,0]
    model_logg = table_model[:,1]
    model_age = np.log10(table_model[:,40])
    model_mass = table_model[:,2]

    f_teff = interpolate.LinearNDInterpolator((model_mass,model_age),
                                              model_T, 
                                              fill_value=np.nan)
    f_logg = interpolate.LinearNDInterpolator((model_mass,model_age),
                                              model_logg, fill_value=np.nan)

    return f_teff,f_logg,model_age,model_mass

def get_isochrone_model(feh,vvcrit):
    '''
    Interpolates MIST isochrones to get a function that gives initial mass
    as a function of main sequence age.
    '''
    file_path = 'Models/MIST/MIST_v1.2_feh_'
    file = file_path + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS_sum.csv'
    table_model = Table.read(file)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = np.log10(table_model['ms_age'])
    
    f_initial_mass = interpolate.interp1d(model_ms_age,model_initial_mass,
                                          fill_value=np.nan)
    
    return f_initial_mass,model_initial_mass,model_ms_age


def model_teff_logg(params,models):
    '''
    Obtains teff and logg from main sequence age and cooling age
    '''
    #Define models to use
    ifmr_model,isochrone_model,cooling_models,fig_name,dist_file_name = models
    f_teff,f_logg,cooling_age_model,final_mass_model = cooling_models
    f_initial_mass,model_initial_mass,ms_age_model = isochrone_model
    
    #Parameters
    ln_ms_age,ln_cooling_age = params
    
    #Sum of main sequence age and cooling age is the total age
    ln_total_age = np.log10(10**ln_cooling_age + 10**ln_ms_age)
    
    #if((10**ln_total_age)/1e9 >= 13.8):
    #    return 1.,1.
    
    #Get the initial mass from the main sequence age using isochrones
    #Return -inf if ms_age values that are not included in the model
    if(np.logical_or(ln_ms_age < np.nanmin(ms_age_model),
                     ln_ms_age > np.nanmax(ms_age_model))):
        return 1.,1.
    initial_mass = f_initial_mass(ln_ms_age)
    
    #Get the final mass from the initial-final mass relation
    #Return -inf if initial_mass values are not included in the model
    '''
    if(ifmr_model == 'Cummings_2018_MIST'):
        if(initial_mass >= max_initial_mass_mist 
           or initial_mass < min_initial_mass_mist):
            return 1.,1.
    elif(ifmr_model == 'Cummings_2018_PARSEC'):
        if(initial_mass >= 8.20 or initial_mass < 0.87):
            return 1.,1.
    elif(ifmr_model == 'Salaris_2009'):
        if(initial_mass < 1.7):
            return 1.,1.
    '''
        
    final_mass = ifmr(initial_mass,ifmr_model)
    
    #Return -inf if the final_mass or the cooling age are not in the 
    #limits of the model
    if(np.logical_or(np.nanmin(final_mass_model) > final_mass,
                     np.nanmax(final_mass_model) < final_mass)):
        return 1.,1.
    if(np.logical_or(np.nanmin(cooling_age_model) > ln_cooling_age,
                     np.nanmax(cooling_age_model) < ln_cooling_age)):
        return 1.,1.
    
    #Get the teff and logg using evolutionary tracs from final mass and 
    #cooling age
    teff_model = f_teff(final_mass,ln_cooling_age)
    logg_model = f_logg(final_mass,ln_cooling_age)
    
    #If both values are nan means that the model doesn't include that value 
    #of final_mass and cooling age. So we do not take into account that point.
    if(np.isnan(teff_model) * np.isnan(logg_model)):
        return 1.,1.
    
    #Saving the likelihoods evaluations
    if(dist_file_name != 'None'):
        save_likelihoods_file = dist_file_name +'.txt'
        save_likelihoods = open(save_likelihoods_file,'a')
        save_likelihoods.write(str(ln_ms_age) + '\t' + 
                               str(ln_cooling_age) + '\t' + 
                               str(ln_total_age) + '\t' + 
                               str(initial_mass) + '\t' + 
                               str(final_mass) + '\n')
    return teff_model,logg_model

def lnlike(params,teff,e_teff,logg,e_logg,models):

    model_teff,model_logg = model_teff_logg(params,models)
    
    if(model_teff == 1. and model_logg==1.):
        return -np.inf
    else:
        loglike_teff_exp = (teff-model_teff)**2/e_teff**2
        loglike_logg_exp = (logg-model_logg)**2/e_logg**2
        return -0.5*(np.sum(loglike_teff_exp + loglike_logg_exp))
    
def ln_posterior_prob(params,teff,e_teff,logg,e_logg,models):
    
    ln_ms_age,ln_cooling_age = params
    ln_ms_age = np.asarray(ln_ms_age)
    
    #if(np.any((10**ln_ms_age)/1e9>13.8)):
    #    return -np.inf
    #elif(np.any((10**ln_cooling_age)/1e9>13.8)):
    #    return -np.inf
    #else:
    return lnlike(params,teff,e_teff,logg,e_logg,models) 
    
    
def ln_prior(params,teff,e_teff,logg,e_logg,models):
    ln_ms_age,ln_cooling_age = params
    ln_ms_age = np.asarray(ln_ms_age)
    
    #if(np.any((10**ln_ms_age)/1e9>13.8)):
    #    return -np.inf
    #elif(np.any((10**ln_cooling_age)/1e9>13.8)):
    #    return -np.inf
    #else:
    
    model_teff,model_logg = model_teff_logg(params,models)
    if(model_teff == 1. and model_logg==1.):
        return -np.inf
    else:
        return 0
    