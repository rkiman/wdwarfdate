import numpy as np
from astropy.table import Table
import os
from .cooling_age import calc_cooling_age,get_cooling_model
from .ifmr import calc_initial_mass
from .ms_age import calc_ms_age,get_isochrone_model
from .bayesian_run_mcmc import run_mcmc, get_initial_conditions
from .extra_func import calc_percentiles, plot_distributions

def calc_wd_age(teff0,e_teff0,logg0,e_logg0,method,
                model_wd='DA',feh='p0.00',vvcrit='0.0',
                model_ifmr = 'Cummings_2018_MIST',
                init_params = [], comparison = [], 
                high_perc = 84, low_perc = 16,
                datatype='yr',
                path='results/',
                nburn_in = 1000, max_n = 100000,
                n_indep_samples = 100,n_mc=2000,
                return_distributions=False,
                plot=False):
    
    """
    Estimates ages for white dwarfs.
    
    Parameters
    ----------
    teff0 : scalar, array. Effective temperature of the white dwarf
    e_teff0 : scalar, array. Error in the effective temperature of the white dwarf
    logg0 : scalar, array. Surface gravity of the white dwarf
    e_logg0 : scalar, arraya. Error in surface gravity of the white dwarf
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
    datatype : string. 'yr', 'Gyr' or 'log'. Units in which the results will be 
               output. Make sure that the comparison array has the same units,
               if you include one.
    path : string. Name of the folder where all the plots and distribution file
           will be save. If it doesn't exist, the code will create it.
    nburn_in : scalar. Number of steps for the burn in. Only useful in 
               Bayesian mode.
    max_n : scalar. Maximum number of steps done by the mcmc to estimate 
            parameters. Only useful in Bayesian mode.
    n_indep_samples : scalar. Number of independent samples. The MCMC will run
                     for n_idep_samples*n_calc_auto_corr steps. Only useful in 
                     Bayesian mode.
    n_mc : scalar. Length of the distribution for each parameter. Only 
           useful in Freq mode.
    return_distributions : True or False. Adds columns to the outputs with the
                           distributions of each parameter. Only useful in 
                           Freq mode.
    plot: True or Flase. If True, plots and saves the figures describing the 
          result in the path given.
    
    Returns
    -------
    results : astropy Table with the results for each white dwarf parameter.
    """
    if(method=='bayesian'):
        #If it doesn't exist, creates a folder to save the plots
        if not os.path.exists(path):
            os.makedirs(path) 

        results = Table(names=('ms_age_median','ms_age_err_low',
                               'ms_age_err_high','cooling_age_median',
                               'cooling_age_err_low','cooling_age_err_high',
                               'total_age_median','total_age_err_low',
                               'total_age_err_high','initial_mass_median',
                               'initial_mass_err_low',
                               'initial_mass_err_high','final_mass_median',
                               'final_mass_err_low','final_mass_err_high'))
        
        if not isinstance(teff0, np.ndarray):
            teff0 = np.array([teff0])
            e_teff0 = np.array([e_teff0])
            logg0 = np.array([logg0])
            e_logg0 = np.array([e_logg0])
        
        if(comparison==[]):
            comparison = [[np.nan] for x in range(len(teff0))]
        for teff0_i,e_teff0_i,logg0_i,e_logg0_i,c_i in zip(teff0,e_teff0,
                                                           logg0,e_logg0,
                                                           comparison):
            print('Running teff:{} logg:{}'.format(teff0_i,logg0_i))
            #Set name of path and wd models to identif results
            wd_path_id = get_wd_path_id(teff0_i,logg0_i,feh,vvcrit,model_wd,
                                        model_ifmr,path) 
            
            #Interpolates models for cooling age and main sequence age
            cooling_models = get_cooling_model(model_wd)
            isochrone_model = get_isochrone_model(feh=feh,vvcrit=vvcrit)
            
            models0 = [model_ifmr,isochrone_model,cooling_models,wd_path_id]
            
            if(init_params==[]):
                init_params = get_initial_conditions(teff0_i,logg0_i,
                                                     model_wd,model_ifmr,
                                                     feh,vvcrit)
            #Check if file exists and remove if it does so if can be filled 
            #again
            if os.path.exists(wd_path_id+'.txt'):
                os.remove(wd_path_id+'.txt')
            
            results_i = calc_bayesian_wd_age(teff0_i,e_teff0_i,
                                             logg0_i,e_logg0_i,
                                             models0, init_params, c_i,
                                             high_perc, low_perc,datatype,
                                             nburn_in,max_n,
                                             n_indep_samples,
                                             plot)
            results.add_row(results_i)
            
    elif(method=='freq'):
        results = calc_wd_age_freq(teff0,e_teff0,logg0,e_logg0,n_mc,
                                   model_wd,feh,vvcrit,model_ifmr,
                                   high_perc,low_perc,datatype,comparison,path,
                                   return_distributions=return_distributions,
                                   plot=plot)
    return results
        

def calc_bayesian_wd_age(teff0,e_teff0,logg0,e_logg0,
                         models0, init_params, comparison,
                         high_perc, low_perc,datatype,
                         nburn_in,max_n,n_indep_samples,plot):
    '''
    Calculates percentiles for main sequence age, cooling age, total age, 
    final mass and initial mass of a white dwarf with teff0 and logg0. 
    Works for one white dwarf at a time. 
    
    comparison: list of results from another paper:
    [log10(ms age (yr)),log10(cooling age (yr)), log10(totla age (yr)), 
    initial mass, final mass]
    '''
    
    _,_,_,wd_path_id = models0
    #Run emcee to obtain likelihood evaluations of ms age, cooling age, 
    #total age, final mass and initial mass
    flat_samples = run_mcmc(teff0, e_teff0, logg0, e_logg0, models0, 
                            init_params, comparison,
                            nburn_in,max_n,n_indep_samples,plot)

    ln_ms_age = flat_samples[:,0]
    ln_cooling_age = flat_samples[:,1]

    #Open file where the likelihood evaluations where saved
    like_eval = np.loadtxt(wd_path_id+'.txt')
    
    #Use the likelihood evaluations for the dependent parameters 
    #and the posterior for the independen parameters
    ln_total_age = like_eval[:,2]
    initial_mass = like_eval[:,3]
    final_mass = like_eval[:,4]
    
    #Calculate percentiles for ms age, cooling age, total age, 
    #initial mass and final mass
    results = calc_percentiles(ln_ms_age, ln_cooling_age, ln_total_age, 
                               initial_mass, final_mass, high_perc, 
                               low_perc, datatype=datatype)
    if(plot==True):            
        if(datatype=='yr'):
            plot_distributions(ln_ms_age, ln_cooling_age, 
                               ln_total_age, initial_mass, final_mass, 
                               datatype, high_perc, low_perc, 
                               comparison = comparison, 
                               name = wd_path_id)
        elif(datatype=='Gyr'):
            plot_distributions((10**ln_ms_age)/1e9, 
                               (10**ln_cooling_age)/1e9, 
                               (10**ln_total_age)/1e9, initial_mass, 
                               final_mass, datatype,
                               high_perc, low_perc, 
                               comparison = comparison, 
                               name = wd_path_id)
        elif(datatype=='log'):
            plot_distributions(ln_ms_age, ln_cooling_age, 
                               ln_total_age, initial_mass, final_mass, 
                               datatype, high_perc, low_perc, 
                               comparison = comparison, 
                               name = wd_path_id)        
    return results

def calc_wd_age_freq(teff0,e_teff0,logg0,e_logg0,n_mc,model_wd,feh,vvcrit,
                     model_ifmr,high_perc,low_perc,datatype,comparison,path,
                     return_distributions,plot):
    '''
    Calculated white dwarfs ages with a frequentist approch. Starts from normal 
    dristribution of teff and logg based on the errors and passes the full
    distribution through the same process to get a distribution of ages.
    '''
    
    if(not isinstance(teff0,np.ndarray)):
        teff0 = np.array([teff0])
        e_teff0 = np.array([e_teff0])
        logg0 = np.array([logg0])
        e_logg0 = np.array([e_logg0])
    
    N = len(teff0)
    
    #Set up the distribution of teff and logg
    teff_dist,logg_dist = [],[]
    for i in range(N):
        if(np.isnan(teff0[i]+e_teff0[i]+logg0[i]+e_logg0[i])):
            teff_dist.append(np.nan)
            logg_dist.append(np.nan)
        else:
            teff_dist.append(np.random.normal(teff0[i],e_teff0[i],n_mc))
            logg_dist.append(np.random.normal(logg0[i],e_logg0[i],n_mc))
    teff_dist,logg_dist = np.array(teff_dist),np.array(logg_dist)
    
    #From teff and logg get ages
    cooling_age_dist,final_mass_dist = calc_cooling_age(teff_dist,logg_dist,
                                                        N,model=model_wd)
    initial_mass_dist = calc_initial_mass(model_ifmr,final_mass_dist)
    ms_age_dist = calc_ms_age(initial_mass_dist,feh=feh,vvcrit=vvcrit)
    total_age_dist = cooling_age_dist + ms_age_dist
    
    #Replace all the ages which are higher than the age of the universe with
    #nans
    mask_nan = np.isnan(total_age_dist)
    total_age_dist[mask_nan] = -1
    
    mask = total_age_dist/1e9 > 13.8
    total_age_dist[mask] = np.nan
    
    total_age_dist[mask_nan] = np.nan

    cooling_age_dist[mask] = np.copy(cooling_age_dist[mask])*np.nan
    final_mass_dist[mask] = np.copy(final_mass_dist[mask])*np.nan
    initial_mass_dist[mask] = np.copy(initial_mass_dist[mask])*np.nan
    ms_age_dist[mask] = np.copy(ms_age_dist[mask])*np.nan
    total_age_dist[mask] = np.copy(total_age_dist[mask])*np.nan
    
    #Calculate percentiles and save results
    results = Table()
    
    median,high_err,low_err = calc_dist_percentiles(final_mass_dist,'none',
                                                    high_perc,low_perc)
    results['final_mass_median'] = median
    results['final_mass_err_high'] = high_err
    results['final_mass_err_low'] = low_err
    
    median,high_err,low_err = calc_dist_percentiles(initial_mass_dist,'none',
                                                    high_perc,low_perc)
    results['initial_mass_median'] = median
    results['initial_mass_err_high'] = high_err
    results['initial_mass_err_low'] = low_err

    median,high_err,low_err = calc_dist_percentiles(cooling_age_dist,datatype,
                                                    high_perc,low_perc) 
    results['cooling_age_median'] = median
    results['cooling_age_err_high'] = high_err
    results['cooling_age_err_low'] = low_err

    median,high_err,low_err = calc_dist_percentiles(ms_age_dist,datatype,
                                                    high_perc,low_perc)    
    results['ms_age_median'] = median
    results['ms_age_err_high'] = high_err
    results['ms_age_err_low'] = low_err
    
    median,high_err,low_err = calc_dist_percentiles(total_age_dist,datatype,
                                                    high_perc,low_perc)    
    results['total_age_median'] = median
    results['total_age_err_high'] = high_err
    results['total_age_err_low'] = low_err
    
    if(return_distributions):
        results['final_mass_dist'] = final_mass_dist
        results['initial_mass_dist'] = initial_mass_dist
        if(datatype=='yr'):
            results['cooling_age_dist'] = cooling_age_dist
            results['ms_age_dist'] = ms_age_dist
            results['total_age_dist'] = total_age_dist
        elif(datatype=='Gyr'):
            results['cooling_age_dist'] = cooling_age_dist/1e9
            results['ms_age_dist'] = ms_age_dist/1e9
            results['total_age_dist'] = total_age_dist/1e9
        elif(datatype=='log'):
            results['cooling_age_dist'] = np.log10(cooling_age_dist)
            results['ms_age_dist'] = np.log10(ms_age_dist)
            results['total_age_dist'] = np.log10(total_age_dist)

    #Plot resulting distributions
    if(plot==True):
        if not os.path.exists(path):
            os.makedirs(path) 
        if(comparison==[]):
            comparison = [[np.nan] for x in range(len(teff0))]
        for x1,x2,x3,x4,x5,x6,x7,x8 in zip(teff0,logg0,ms_age_dist,
                                           cooling_age_dist,total_age_dist,
                                           initial_mass_dist,final_mass_dist,
                                           comparison):
            wd_path_id = get_wd_path_id(x1,x2,feh,vvcrit,model_wd,
                                        model_ifmr,path) 
            if(datatype=='yr'):
                plot_distributions(x3,x4,x5,
                                   x6,x7,high_perc, low_perc, datatype,
                                   comparison=x8, name = wd_path_id + '_freq')
            elif(datatype=='Gyr'):
                plot_distributions(x3/1e9,x4/1e9,x5/1e9,
                                   x6,x7,high_perc, low_perc, datatype,
                                   comparison=x8, name = wd_path_id + '_freq')
            elif(datatype=='log'):
                plot_distributions(np.log10(x3),np.log10(x4),
                                   np.log10(x5),
                                   x6,x7,high_perc, low_perc, datatype,
                                   comparison=x8, name = wd_path_id + '_freq')
    
    return results

def calc_dist_percentiles(dist,datatype,high_perc,low_perc):
    if(datatype=='yr' or datatype=='none'):
        median = np.array([np.nanpercentile(x,50) for x in dist])
        h=[np.nanpercentile(x,high_perc)-np.nanpercentile(x,50) for x in dist]
        l=[np.nanpercentile(x,50)-np.nanpercentile(x,low_perc) for x in dist]
        h,l = np.array(h),np.array(l)
    elif(datatype=='Gyr'):
        dist1 = dist/1e9
        median = np.array([np.nanpercentile(x,50) for x in dist1])
        h=[np.nanpercentile(x,high_perc)-np.nanpercentile(x,50) for x in dist1]
        l=[np.nanpercentile(x,50)-np.nanpercentile(x,low_perc) for x in dist1]
        h,l = np.array(h),np.array(l)
    elif(datatype=='log'):
        dist1 = np.log10(dist)
        median = np.array([np.nanpercentile(x,50) for x in dist1])
        h=[np.nanpercentile(x,high_perc)-np.nanpercentile(x,50) for x in dist1]
        l=[np.nanpercentile(x,50)-np.nanpercentile(x,low_perc) for x in dist1]
        h,l = np.array(h),np.array(l)
    return median,h,l

def get_wd_path_id(teff0,logg0,feh,vvcrit,model_wd,model_ifmr,path):
    #Set the name to identify the results from each white dwarf
    teff_logg_name = 'teff_' + str(teff0) + '_logg_' + str(logg0)
    models_name_mist = '_feh_' + feh + '_vvcrit_' + vvcrit
    models_name = models_name_mist  + '_' + model_wd + '_' + model_ifmr
    return path + teff_logg_name + models_name
