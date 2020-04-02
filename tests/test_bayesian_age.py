#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.bayesian_age import ln_posterior_prob, get_isochrone_model
from wdwarfdate.bayesian_age import get_cooling_model
import numpy as np

def test_ln_posterior_prob():
    ln_ms_age0 = np.log10(1181155122.7574291)
    ln_cooling_age0 = np.log10(1911395787.679186)
    teff0 = 6510
    e_teff0 = 170 
    logg0 = 8.01 
    e_logg0 = 0.2
    
    cooling_models = get_cooling_model('DA')
    ifmr_model = 'Cummings_2018_MIST'
    isochrone_model = get_isochrone_model(feh='p0.00',vvcrit='0.0')
    
    fig_name = 'run_' 
    dist_file_name = 'None'

    models0 = [ifmr_model,isochrone_model,cooling_models,fig_name,
               dist_file_name]
    params = ln_ms_age0,ln_cooling_age0,0
    params_off = 8.,ln_cooling_age0,0
    
    post_true = ln_posterior_prob(params,teff0,e_teff0,logg0,e_logg0,models0)
    post_off = ln_posterior_prob(params_off,teff0,e_teff0,logg0,e_logg0,
                                 models0)
    
    assert(post_true > post_off)
    
def test_get_cooling_model():
    
    #Data from http://www.montrealwhitedwarfdatabase.org/evolution.html
    teff = 4670.609661
    logg = 7.794246000000001
    final_mass = 0.451
    cooling_age = np.log10(4.440*1e9)
    
    #Define model with the same data base
    f_teff,f_logg,model_age,model_mass = get_cooling_model('DA')
    
    assert(np.isclose(f_teff(final_mass,cooling_age),teff,atol=120))
    assert(np.isclose(f_logg(final_mass,cooling_age),logg,atol=0.05))
    

