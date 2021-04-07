#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .ifmr import ifmr_bayesian

min_initial_mass_mist = 0.83 - 0.1
max_initial_mass_mist =  7.20 + 0.1

log_age_universe = 15 #log10(20*1e9)
#10.13988 #log10(13.8*1e9)

def model_teff_logg(params,models):
    '''
    Obtains teff and logg from main sequence age and cooling age
    '''
    #Define models to use
    model_ifmr,isochrone_model,cooling_models,f=models
    f_teff,f_logg,cooling_age_model,final_mass_model = cooling_models
    f_initial_mass,model_initial_mass,ms_age_model = isochrone_model
    
    #Parameters
    ln_ms_age, ln_cooling_age, delta_m = params
    
    #Sum of main sequence age and cooling age is the total age
    ln_total_age = np.log10(10**ln_cooling_age + 10**ln_ms_age)
    #print('total age: {}'.format(ln_total_age))
    
    #if(ln_total_age >= log_age_universe):
    #    return 1.,1.
    
    #Get the initial mass from the main sequence age using isochrones
    #Return -inf if ms_age values that are not included in the model
    #if(np.logical_or(ln_ms_age < np.nanmin(ms_age_model),
    #                 ln_ms_age >= log_age_universe)):
    if(np.logical_or(ln_ms_age < np.nanmin(ms_age_model),
                     ln_ms_age > np.nanmax(ms_age_model))):
        #print('ms age out of range')
        return 1.,1.
    initial_mass = f_initial_mass(ln_ms_age)
    #print('initial mass: {}'.format(initial_mass))
    
    #Get the final mass from the initial-final mass relation
    #Return -inf if initial_mass values are not included in the model
    
    if(model_ifmr == 'Cummings_2018_MIST'):
        if(initial_mass >= max_initial_mass_mist 
           or initial_mass < min_initial_mass_mist):
            return 1.,1.
    elif(model_ifmr == 'Cummings_2018_PARSEC'):
        if(initial_mass >= 8.20 or initial_mass < 0.87):
            return 1.,1.
    elif(model_ifmr == 'Salaris_2009'):
        if(initial_mass < 1.7):
            return 1.,1.
    
    
    final_mass = ifmr_bayesian(initial_mass,model_ifmr,min_initial_mass_mist,
                               max_initial_mass_mist)
    final_mass = final_mass + delta_m
    #print('final mass: {}'.format(final_mass))
    
    #Return -inf if the final_mass or the cooling age are not in the 
    #limits of the model
    if(np.logical_or(np.nanmin(final_mass_model) > final_mass,
                     np.nanmax(final_mass_model) < final_mass)):
        #print('final mass out of range')
        return 1.,1.
    if(np.logical_or(np.nanmin(cooling_age_model) > ln_cooling_age,
                     np.nanmax(cooling_age_model) < ln_cooling_age)):
        #print('cooling age out of range')
        return 1.,1.
    
    #Get the teff and logg using evolutionary tracs from final mass and 
    #cooling age
    teff_model = f_teff(final_mass,ln_cooling_age)
    logg_model = f_logg(final_mass,ln_cooling_age)
    
    #If both values are nan means that the model doesn't include that value 
    #of final_mass and cooling age. So we do not take into account that point.
    if(np.isnan(teff_model+logg_model)):
        #print('teff or logg nan')
        return 1.,1.
    elif(np.logical_or(teff_model<0,logg_model<0)):
        return 1.,1.
    
    #Saving the likelihoods evaluations
    if(type(f)!=str):
        f.write(str(ln_ms_age) + '\t' + 
                str(ln_cooling_age) + '\t' + 
                str(ln_total_age) + '\t' + 
                str(initial_mass) + '\t' + 
                str(final_mass) + '\n')
    return teff_model,logg_model

def lnlike(params,teff,e_teff,logg,e_logg,models):
    
    sigma_m = 0.08
    _,_, delta_m = params
    model_teff,model_logg = model_teff_logg(params,models)
    
    if(model_teff == 1. and model_logg==1.):
        return -np.inf
    else:
        loglike_teff_exp = (teff-model_teff)**2/e_teff**2
        loglike_logg_exp = (logg-model_logg)**2/e_logg**2
        loglike_delta_m = delta_m**2/sigma_m**2
        return (-.5*(np.sum(loglike_teff_exp+loglike_logg_exp+loglike_delta_m))) 
    
def ln_posterior_prob(params,teff,e_teff,logg,e_logg,models):
    
    ln_ms_age, ln_cooling_age, delta_m = params
    ln_ms_age = np.asarray(ln_ms_age)
    
    #if(np.any((10**ln_ms_age)/1e9>13.8)):
    #    return -np.inf
    #elif(np.any((10**ln_cooling_age)/1e9>13.8)):
    #    return -np.inf
    #else:
    return lnlike(params,teff,e_teff,logg,e_logg,models) 
    
    
def ln_prior(params,teff,e_teff,logg,e_logg,models):
    ln_ms_age,ln_cooling_age, delta_m = params
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
    