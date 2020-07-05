import numpy as np
from astropy.table import Table
import os
from .cooling_age import calc_cooling_age
from .ifmr import calc_initial_mass
from .ms_age import calc_ms_age
from .bayesian_age import get_cooling_model, get_isochrone_model
from .bayesian_run_mcmc import run_mcmc
from .extra_func import calc_percentiles, plot_distributions

def calc_wd_age(teff0,e_teff0,logg0,e_logg0,method,
                model_wd='DA',feh='p0.00',vvcrit='0.0',
                model_ifmr = 'Cummings_2018_MIST',
                init_params = [], comparison = [], 
                high_perc = 84, low_perc = 16,
                datatype='yr',
                path='results/',
                nburn_in = 1000,n_calc_auto_corr = 10000,
                n_idep_samples = 1000,n_mc=2000,
                return_distributions=False):
    
    """
    Estimates ages for white dwarfs.
    
    Parameters
    ----------
    teff0 : scalar. Effective temperature of the white dwarf
    e_teff0 : scalar. Error in the effective temperature of the white dwarf
    logg0 : scalar. Surface gravity of the white dwarf
    e_logg0 : scalar. Error in surface gravity of the white dwarf
    method : string. 'bayesian' or 'freq'. Bayesian will run an mcmc and 
             output the distributions. Freq runs a normal distribution 
             centered at the value with a std of the error through all the 
             models chosen.
    model_wd : string. Spectral type of the white dwarf 'DA' or 'DB'. 
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00' 
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'
    model_ifmr : string. Initial to final mass relation model. Can be 
                 'Cummings_2018_MIST', 'Cummings_2018_PARSEC' 
                 or 'Salaris_2009'.
    init_params : list, array. Optional initial parameter for the burn in of 
                  the mcmc for:
                  [log10 ms age, log10 cooling age, delta m]. 
                  Only useful in Bayesian mode. 
    comparison : list, array. Values to compare the results for log10 ms age, 
                 log10 cooling age, delta m.
    high_perc : scalar. Percentage at which the high errors will be calculated. 
    low_perc : scalar. Percentage at which the low errors will be calculated. 
    datatype : string. 'yr' or 'Gyr'. Units in which the results will be 
               output.
    path : string. Name of the folder where all the plots and distribution file
           will be save. If it doesn't exist, the code will create it.
    nburn_in : scalar. Number of steps for the burn in. Only useful in 
               Bayesian mode.
    n_calc_auto_corr : scalar. Number of steps taken to calculate 
                       auto-correlation time. Only useful in Bayesian mode.
    n_idep_samples : scalar. Number of independent samples. The MCMC will run
                     for n_idep_samples*n_calc_auto_corr steps. Only useful in 
                     Bayesian mode.
    n_mc : scalar. Length of the distribution for each parameter. Only 
           useful in Freq mode.
    return_distributions : True or False. Adds columns to the outputs with the
                           distributions of each parameter. Only useful in 
                           Freq mode.
    Returns
    -------
    results : astropy Table with the results for each white dwarf parameter.
    """
    if(method=='bayesian'):
        #Set name of path and wd models to identif results
        wd_path_id = get_wd_path_id(teff0,logg0,feh,vvcrit,model_wd,model_ifmr,
                                    path) 
        
        #Interpolates models for cooling age and main sequence age
        cooling_models = get_cooling_model(model_wd)
        isochrone_model = get_isochrone_model(feh=feh,vvcrit=vvcrit)
        
        models0 = [model_ifmr,isochrone_model,cooling_models,wd_path_id]
        
        #If it doesn't exist, creates a folder to save the plots
        if not os.path.exists(path):
            os.makedirs(path)           
        #Check if file exists and remove if it does so if can be filled again
        if os.path.exists(wd_path_id+'.txt'):
            os.remove(wd_path_id+'.txt')
        
        results = calc_bayesian_wd_age(teff0,e_teff0,logg0,e_logg0,
                                       models0, init_params, comparison,
                                       high_perc, low_perc,datatype,
                                       nburn_in,n_calc_auto_corr,
                                       n_idep_samples)
    elif(method=='freq'):
        results = calc_wd_age_freq(teff0,e_teff0,logg0,e_logg0,n_mc,
                                   model_wd,feh,vvcrit,
                                   model_ifmr,
                                   return_distributions=return_distributions)
    return results
        

def calc_bayesian_wd_age(teff0,e_teff0,logg0,e_logg0,
                         models0, init_params, comparison,
                         high_perc, low_perc,datatype,
                         nburn_in,n_calc_auto_corr,n_idep_samples):
    '''
    Calculates percentiles for main sequence age, cooling age, total age, 
    final mass and initial mass of a white dwarf with teff0 and logg0. 
    Works for one white dwarf at a time. 
    
    comparison: list of results from another paper:
    [log10(ms age (yr)),log10(cooling age (yr)), log10(totla age (yr)), 
    initial mass, final mass]
    '''
        
    #Run emcee to obtain likelihood evaluations of ms age, cooling age, 
    #total age, final mass and initial mass
    flat_samples = run_mcmc(teff0, e_teff0, logg0, e_logg0, models0, 
                            init_params, comparison,
                            nburn_in=nburn_in,
                            n_calc_auto_corr=n_calc_auto_corr,
                            n_idep_samples=n_idep_samples)

    ln_ms_age = flat_samples[:,0]
    ln_cooling_age = flat_samples[:,1]

    model_ifmr,isochrone_model,cooling_models,wd_path_id = models0
    #Open file where the likelihood evaluations where saved
    like_eval = np.loadtxt(wd_path_id+'.txt')
    
    #Use the likelihood evaluations for the dependent parameters 
    #and the posterior for the independen parameters
    ln_total_age = like_eval[500:,2]
    initial_mass = like_eval[500:,3]
    final_mass = like_eval[500:,4]
        
    #Calculate percentiles for ms age, cooling age, total age, 
    #initial mass and final mass
    results = calc_percentiles(ln_ms_age, ln_cooling_age, ln_total_age, 
                               initial_mass, final_mass, high_perc, 
                               low_perc, datatype=datatype)
    
    plot_distributions(teff0, logg0, ln_ms_age, ln_cooling_age, 
                       ln_total_age, initial_mass, final_mass, 
                       high_perc, low_perc, 
                       comparison = comparison, 
                       name = wd_path_id)

    return results

def calc_wd_age_freq(teff,e_teff,logg,e_logg,n_mc=2000,
                     model_wd='DA',feh='p0.00',vvcrit='0.0',
                     model_ifmr = 'Cummings_2018_MIST',
                     return_distributions=False):
    '''
    Calculated white dwarfs ages with a frequentist approch. Starts from normal 
    dristribution of teff and logg based on the errors and passes the full
    distribution through the same process to get a distribution of ages.
    '''
    
    if(not isinstance(teff,np.ndarray)):
        teff = np.array([teff])
        e_teff = np.array([e_teff])
        logg = np.array([logg])
        e_logg = np.array([e_logg])
    
    N = len(teff)
    
    teff_dist,logg_dist = [],[]
    
    for i in range(N):
        if(np.isnan(teff[i]+e_teff[i]+logg[i]+e_logg[i])):
            teff_dist.append(np.nan)
            logg_dist.append(np.nan)
        else:
            teff_dist.append(np.random.normal(teff[i],e_teff[i],n_mc))
            logg_dist.append(np.random.normal(logg[i],e_logg[i],n_mc))
    teff_dist,logg_dist = np.array(teff_dist),np.array(logg_dist)
        
    cooling_age_dist,final_mass_dist = calc_cooling_age(teff_dist,logg_dist,
                                                        n_mc,N,model=model_wd)
    
    initial_mass_dist = calc_initial_mass(model_ifmr,final_mass_dist,n_mc)
    
    ms_age_dist = calc_ms_age(initial_mass_dist,feh=feh,vvcrit=vvcrit)
    
    total_age_dist = cooling_age_dist + ms_age_dist
    
    mask = np.logical_or(np.logical_or(ms_age_dist/1e9 > 13.8,
                                       total_age_dist/1e9 > 13.8),
                         cooling_age_dist/1e9 > 13.8)
    
    cooling_age_dist[mask] = np.copy(cooling_age_dist[mask])*np.nan
    final_mass_dist[mask] = np.copy(final_mass_dist[mask])*np.nan
    initial_mass_dist[mask] = np.copy(initial_mass_dist[mask])*np.nan
    ms_age_dist[mask] = np.copy(ms_age_dist[mask])*np.nan
    total_age_dist[mask] = np.copy(total_age_dist[mask])*np.nan
    
    results = Table()
    
    median,high_err,low_err = calc_dist_percentiles(final_mass_dist)
    results['final_mass_median'] = median
    results['final_mass_err_high'] = high_err
    results['final_mass_err_low'] = low_err
    
    median,high_err,low_err = calc_dist_percentiles(initial_mass_dist)
    results['initial_mass_median'] = median
    results['initial_mass_err_high'] = high_err
    results['initial_mass_err_low'] = low_err

    median,high_err,low_err = calc_dist_percentiles(cooling_age_dist)    
    results['cooling_age_median'] = median
    results['cooling_age_err_high'] = high_err
    results['cooling_age_err_low'] = low_err

    median,high_err,low_err = calc_dist_percentiles(ms_age_dist)    
    results['ms_age_median'] = median
    results['ms_age_err_high'] = high_err
    results['ms_age_err_low'] = low_err
    
    median,high_err,low_err = calc_dist_percentiles(total_age_dist)    
    results['total_age_median'] = median
    results['total_age_err_high'] = high_err
    results['total_age_err_low'] = low_err
    
    if(return_distributions):
        results['final_mass_dist'] = final_mass_dist
        results['initial_mass_dist'] = initial_mass_dist
        results['cooling_age_dist'] = cooling_age_dist
        results['ms_age_dist'] = ms_age_dist
        results['total_age_dist'] = total_age_dist
    return results

def calc_dist_percentiles(dist):
    median = np.array([np.nanpercentile(x,50) for x in dist])
    h=np.array([np.nanpercentile(x,84.1)-np.nanpercentile(x,50) for x in dist])
    l=np.array([np.nanpercentile(x,50)-np.nanpercentile(x,15.9) for x in dist])
    return median,h,l

def get_wd_path_id(teff0,logg0,feh,vvcrit,model_wd,model_ifmr,path):
    #Set the name to identify the results from each white dwarf
    teff_logg_name = 'teff_' + str(teff0) + '_logg_' + str(logg0)
    models_name_mist = '_feh_' + feh + '_vvcrit_' + vvcrit
    models_name = models_name_mist  + '_' + model_wd + '_' + model_ifmr
    return path + teff_logg_name + models_name
