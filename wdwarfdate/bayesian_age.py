#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table
from scipy import interpolate

def ifmr(initial_mass):
    '''
    Define Initial-Final Mass relation
    '''
    #Initialize variables
    initial_mass = np.asarray(initial_mass)
    final_mass = np.copy(initial_mass)*np.nan
    final_mass = np.asarray(final_mass)
    
    #Initial-Final mass relation from Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
    mask1 = (0.83 <= initial_mass) * (initial_mass < 2.85)
    mask2 = (2.85 <= initial_mass) * (initial_mass < 3.60)
    mask3 = (3.60 <= initial_mass) * (initial_mass < 7.20)
    final_mass[mask1] = initial_mass[mask1] * 0.08 + 0.489
    final_mass[mask2] = initial_mass[mask2] * 0.187 + 0.184
    final_mass[mask3] = initial_mass[mask3] * 0.107 + 0.471
    
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

    f_teff = interpolate.LinearNDInterpolator((model_mass,model_age),model_T, fill_value=np.nan)
    f_logg = interpolate.LinearNDInterpolator((model_mass,model_age),model_logg, fill_value=np.nan)

    return f_teff,f_logg,model_age,model_mass,model_wd

def get_isochrone_model(feh,vvcrit):
    '''
    Interpolates MIST isochrones to get a function that gives initial mass
    as a function of main sequence age.
    '''
    file = 'Models/MIST/MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_EEPS_sum.csv'
    table_model = Table.read(file)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = np.log10(table_model['ms_age'])
    
    f_initial_mass = interpolate.interp1d(model_ms_age,model_initial_mass, fill_value=np.nan)
    name = 'feh_' + feh + '_vvcrit_' + vvcrit
    return f_initial_mass,model_initial_mass,model_ms_age,name


def model_teff_logg(params,models,name_file):
    '''
    Obtains teff and logg from main sequence age and cooling age
    '''
    #Define models to use
    ifmr_model,isochrone_model,cooling_models = models
    f_teff,f_logg,cooling_age_model,final_mass_model,model_wd = cooling_models
    f_initial_mass,model_initial_mass,ms_age_model,model_name = isochrone_model
    
    #Parameters
    ln_ms_age,ln_cooling_age = params
    
    #print('cooling age: {}'.format(10**ln_cooling_age))
    
    #Sum of main sequence age and cooling age is the total age
    ln_total_age = np.log10(10**ln_cooling_age + 10**ln_ms_age)
    
    if((10**ln_total_age)/1e9 >= 13.8):
        return 1.,1.
    
    #Get the initial mass from the main sequence age using isochrones
    #Return -inf if ms_age values that are not included in the model
    if(np.logical_or(ln_ms_age < np.nanmin(ms_age_model),ln_ms_age > np.nanmax(ms_age_model))):
        return 1.,1.
    initial_mass = f_initial_mass(ln_ms_age)
    
    #print('initial_mass: {}'.format(initial_mass))
    
    #Get the final mass from the initial-final mass relation
    #Return -inf if initial_mass values are not included in the model
    if(initial_mass >= 7.20 or initial_mass < 0.83):
        return 1.,1.
    final_mass = ifmr(initial_mass)
    
    #print('final_mass: {}'.format(final_mass))
    
    #Return -inf if the final_mass or the cooling age are not in the limits of the model
    if(np.logical_or(np.nanmin(final_mass_model) > final_mass,np.nanmax(final_mass_model) < final_mass)):
        return 1.,1.
    if(np.logical_or(np.nanmin(cooling_age_model) > ln_cooling_age,np.nanmax(cooling_age_model) < ln_cooling_age)):
        return 1.,1.
    
    #Get the teff and logg using evolutionary tracs from final mass and cooling age
    teff_model,logg_model = f_teff(final_mass,ln_cooling_age),f_logg(final_mass,ln_cooling_age)
    #print('model_teff:{} model_logg:{}'.format(teff_model,logg_model))
    
    #If both values are nan means that the model doesn't include that value of final_mass and
    #cooling age. So we do not take into account that point.
    if(np.isnan(teff_model) * np.isnan(logg_model)):
        return 1.,1.
    
    #Saving the likelihoods evaluations
    save_likelihoods_file = name_file + '_' + model_name + '_' + model_wd + '_' + ifmr_model +'.txt'
    save_likelihoods = open(save_likelihoods_file,'a')
    save_likelihoods.write(str(ln_ms_age) + '\t' + str(ln_cooling_age) + '\t' + str(ln_total_age) + '\t' + 
                           str(initial_mass) + '\t' + str(final_mass) + '\n')
    return teff_model,logg_model

def lnlike(params,teff,e_teff,logg,e_logg,models):
    name_file = 'teff_{}_logg_{}'.format(teff,logg)
    model_teff,model_logg = model_teff_logg(params,models,name_file)
    if(model_teff == 1. and model_logg==1.):
        return -np.inf
    else:
        return -0.5*(np.sum((teff-model_teff)**2/e_teff**2 + (logg-model_logg)**2/e_logg**2))
    
def ln_posterior_prob(params,teff,e_teff,logg,e_logg,models):
    
    ln_ms_age,ln_cooling_age = params
    ln_ms_age = np.asarray(ln_ms_age)
    
    if(np.any((10**ln_ms_age)/1e9>13.8)):
        return -np.inf
    elif(np.any((10**ln_cooling_age)/1e9>13.8)):
        return -np.inf
    else:
        return lnlike(params,teff,e_teff,logg,e_logg,models) 
    
    
def ln_prior(params,teff,e_teff,logg,e_logg,models):
    ln_ms_age,ln_cooling_age = params
    ln_ms_age = np.asarray(ln_ms_age)
    
    if(np.any((10**ln_ms_age)/1e9>13.8)):
        return -np.inf
    elif(np.any((10**ln_cooling_age)/1e9>13.8)):
        return -np.inf
    else:
        name_file = 'teff_{}_logg_{}'.format(teff,logg)
        model_teff,model_logg = model_teff_logg(params,models,name_file)
        if(model_teff == 1. and model_logg==1.):
            return -np.inf
        else:
            return 0
    