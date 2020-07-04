import numpy as np
from astropy.table import Table
import os
from .coolingage import calc_cooling_age
from .final2initial_mass import calc_initial_mass
from .ms_age import calc_ms_age
from .bayesian_age import get_cooling_model, get_isochrone_model
from .bayesian_run_mcmc import run_mcmc
from .extra_func import calc_percentiles, plot_distributions

def calc_bayesian_wd_age(teff0,e_teff0,logg0,e_logg0,n_mc=1000,
                         model_wd='DA',feh='p0.00',vvcrit='0.0',
                         model_ifmr = 'Cummings_2018_MIST',
                         init_params = [], comparison = [], n = 500, 
                         high_perc = 84, low_perc = 16,
                         plot = True, save_dist = True, datatype='Gyr',
                         path='results/'):
    '''
    Calculates percentiles for main sequence age, cooling age, total age, 
    final mass and initial mass of a white dwarf with teff0 and logg0. 
    Works for one white dwarf at a time. 
    
    comparison: list of results from another paper:
    [log10(ms age (yr)),log10(cooling age (yr)), log10(totla age (yr)), 
    initial mass, final mass]
    '''

    #Set the name to identify the results from each white dwarf
    teff_logg_name = 'teff_' + str(teff0) + '_logg_' + str(logg0)
    models_name_mist = '_feh_' + feh + '_vvcrit_' + vvcrit
    models_name = models_name_mist  + '_' + model_wd + '_' + model_ifmr
    file_like_eval =  path + teff_logg_name + models_name + '.txt'
    fig_name = path + teff_logg_name + models_name
    
    if(save_dist == True):
        dist_file_name = path + teff_logg_name + models_name
    elif(save_dist == False):
        dist_file_name = 'None'
    
    #Interpolates models for cooling age and main sequence age
    cooling_models = get_cooling_model(model_wd)
    isochrone_model = get_isochrone_model(feh=feh,vvcrit=vvcrit)
    
    models0 = [model_ifmr,isochrone_model,cooling_models,fig_name,
               dist_file_name]
    
    #If it doesn't exist, creates a folder to save the plots
    if not os.path.exists(path):
        os.makedirs(path)
       
    #Check if file exists and remove if it does so if can be filled again
    if os.path.exists(file_like_eval):
        os.remove(file_like_eval)
        
    #Run emcee to obtain likelihood evaluations of ms age, cooling age, 
    #total age, final mass and initial mass
    flat_samples = run_mcmc(teff0, e_teff0, logg0, e_logg0, models0, 
                            init_params = init_params, 
                            n=n, nsteps=n_mc, plot=plot, 
                            figname = fig_name, comparison=comparison)

    ln_ms_age = flat_samples[:,0]
    ln_cooling_age = flat_samples[:,1]
    if(save_dist == True):
        #Open file where the likelihood evaluations where saved
        like_eval = np.loadtxt(file_like_eval)
        
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
                           name = path + teff_logg_name + models_name)
    else:
        results = calc_percentiles(ln_ms_age, ln_cooling_age, [], [], 
                                   [], high_perc, low_perc, datatype=datatype)
        
    return results

def calc_wd_age(teff,e_teff,logg,e_logg,n_mc=2000,
                model_wd='DA',feh='p0.00',vvcrit='0.0',
                model_ifmr = 'Cummings_2018',
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
    
    results['final_mass_median'] = np.array([np.nanpercentile(x,50) for x in final_mass_dist])
    results['final_mass_err_high'] = np.array([np.nanpercentile(x,84.1345)-np.nanpercentile(x,50) for x in final_mass_dist])
    results['final_mass_err_low'] = np.array([np.nanpercentile(x,50)-np.nanpercentile(x,15.8655) for x in final_mass_dist])
    
    results['initial_mass_median'] = np.array([np.nanpercentile(x,50) for x in initial_mass_dist])
    results['initial_mass_err_high'] = np.array([np.nanpercentile(x,84.1345)-np.nanpercentile(x,50) for x in initial_mass_dist])
    results['initial_mass_err_low'] = np.array([np.nanpercentile(x,50)-np.nanpercentile(x,15.8655) for x in initial_mass_dist])

    results['cooling_age_median'] = np.array([np.nanpercentile(x,50) for x in cooling_age_dist])
    results['cooling_age_err_high'] = np.array([np.nanpercentile(x,84.1345)-np.nanpercentile(x,50) for x in cooling_age_dist])
    results['cooling_age_err_low'] = np.array([np.nanpercentile(x,50)-np.nanpercentile(x,15.8655) for x in cooling_age_dist])

    results['ms_age_median'] = np.array([np.nanpercentile(x,50) for x in ms_age_dist])
    results['ms_age_err_high'] = np.array([np.nanpercentile(x,84.1345)-np.nanpercentile(x,50) for x in ms_age_dist])
    results['ms_age_err_low'] = np.array([np.nanpercentile(x,50)-np.nanpercentile(x,15.8655) for x in ms_age_dist])

    results['total_age_median'] = np.array([np.nanpercentile(x,50) for x in total_age_dist])
    results['total_age_err_high'] = np.array([np.nanpercentile(x,84.1345)-np.nanpercentile(x,50) for x in total_age_dist])
    results['total_age_err_low'] = np.array([np.nanpercentile(x,50)-np.nanpercentile(x,15.8655) for x in total_age_dist])    
    
    if(return_distributions):
        results['final_mass_dist'] = final_mass_dist
        results['initial_mass_dist'] = initial_mass_dist
        results['cooling_age_dist'] = cooling_age_dist
        results['ms_age_dist'] = ms_age_dist
        results['total_age_dist'] = total_age_dist
    return results
