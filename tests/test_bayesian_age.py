#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.bayesian_age import ln_posterior_prob, get_isochrone_model, get_cooling_model
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
    
    fig_name = 'test_' 
    
    models0 = [ifmr_model,isochrone_model,cooling_models,fig_name]
    params = ln_ms_age0,ln_cooling_age0
    params_off = 8.,ln_cooling_age0
    
    post_true = ln_posterior_prob(params,teff0,e_teff0,logg0,e_logg0,models0)
    post_off = ln_posterior_prob(params_off,teff0,e_teff0,logg0,e_logg0,models0)
    
    assert(post_true > post_off)

