import numpy as np
from astropy.table import Table
import os
from .coolingage import calc_cooling_age
from .final2initial_mass import calc_initial_mass
from .ms_age import calc_ms_age
from .bayesian_age import get_cooling_model, get_isochrone_model
from .bayesian_results import run_mcmc
from .extra_func import calc_percentiles, plot_distributions

def calc_bayesian_wd_age(teff0,e_teff0,logg0,e_logg0,n_mc=1000,
                         model_wd='DA',feh='p0.00',vvcrit='0.0',
                         model_ifmr = 'Cummings_2018_MIST',
                         init_params = [], comparison = [], n = 500, 
                         high_perc = 84, low_perc = 16,
                         plot = True, save_dist = True, datatype='Gyr'):
    '''
    Calculates percentiles for main sequence age, cooling age, total age, 
    final mass and initial mass of a white dwarf with teff0 and logg0. Works for
    one white dwarf at a time. 
    
    comparison: list of results from another paper:
    [log10(ms age (yr)),log10(cooling age (yr)), log10(totla age (yr)), 
    initial mass, final mass]
    '''

    #Set the name to identify the results from each white dwarf
    teff_logg_name = 'results/teff_' + str(teff0) + '_logg_' + str(logg0)
    models_name =  '_feh_' + feh + '_vvcrit_' + vvcrit + '_' + model_wd + '_' + model_ifmr
    file_like_eval =  teff_logg_name + models_name + '.txt'
    fig_name = teff_logg_name + models_name
    
    if(save_dist == True):
        dist_file_name = teff_logg_name + models_name
    elif(save_dist == False):
        dist_file_name = 'None'
    
    #Interpolates models for cooling age and main sequence age
    cooling_models = get_cooling_model(model_wd)
    isochrone_model = get_isochrone_model(feh=feh,vvcrit=vvcrit)
    
    models0 = [model_ifmr,isochrone_model,cooling_models,fig_name,
               dist_file_name]
    
    #If it doesn't exist, creates a folder to save the plots
    if not os.path.exists('results'):
        os.makedirs('results')
       
    #Check if file exists and remove if it does so if can be filled again
    if os.path.exists(file_like_eval):
        os.remove(file_like_eval)
        
    #Run emcee to obtain likelihood evaluations of ms age, cooling age, total age,
    #final mass and initial mass
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
        
        #Calculate percentiles for ms age, cooling age, total age, initial mass and final mass
        results = calc_percentiles(ln_ms_age, ln_cooling_age, ln_total_age, initial_mass, 
                                   final_mass, high_perc, low_perc, datatype=datatype)
        
        plot_distributions(teff0, logg0, ln_ms_age, ln_cooling_age, ln_total_age, 
                           initial_mass, final_mass, high_perc, low_perc, 
                           comparison = comparison, name = teff_logg_name + models_name)
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
    
    mask = np.logical_or(np.logical_or(ms_age_dist/1e9 > 13.8,total_age_dist/1e9 > 13.8),cooling_age_dist/1e9 > 13.8)
    
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

'''
def plot_comparison(results,final_mass_compare,intial_mass_compare,cooling_age_compare,
                    ms_age_compare,total_age_compare,scale_age):
    result_total_age
    result_total_age_err_low
    result_total_age_err_high
    color_line = 'k'
    lw = .5
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,6))
    
    ax1.errorbar(total_age_compare,result_total_age,
                 xerr=(garces2011['e_Age_lo'],garces2011['e_Age_hi']),
                 yerr=(result_garces2011['total_age_err_low']/1e9,result_garces2011['total_age_err_high']/1e9), fmt='.')
    x = np.linspace(0,10)
    ax1.plot(x,x,'-',color=color_line,linewidth=lw)
    ax1.set_xlabel('Total Age')
    ax1.tick_params('both',direction='in',top=True,right=True)
    ax1.tick_params('y',which='minor',direction='in',right=True)
    
    
    ax2.errorbar(garces2011['Minitial_mass'],result_garces2011['initial_mass_median'],
                 xerr=garces2011['e_initial_mass'],
                 yerr=(result_garces2011['initial_mass_err_low'],result_garces2011['initial_mass_err_high']),
                 fmt='.')
    x = np.linspace(0.5,4)
    ax2.plot(x,x,'-',color=color_line,linewidth=lw)
    ax2.set_xlabel('Initial Mass')
    ax2.tick_params('both',direction='in',top=True,right=True)
    ax2.tick_params('y',which='minor',direction='in',right=True)
    
    ax3.errorbar(garces2011['tcool'],result_garces2011['cooling_age_median']/1e9,
                 xerr=garces2011['e_tcool'],
                 yerr=(result_garces2011['cooling_age_err_low']/1e9,result_garces2011['cooling_age_err_high']/1e9), fmt='.')
    x = np.linspace(0,6)
    ax3.plot(x,x,'-',color=color_line,linewidth=lw)
    ax3.set_xlabel('Cooling Age')
    ax3.tick_params('both',direction='in',top=True,right=True)
    ax3.tick_params('y',which='minor',direction='in',right=True)
    
    ax4.errorbar(garces2011['ms_age'],result_garces2011['ms_age_median']/1e9,
                 xerr=(garces2011['e_ms_age_lo'],garces2011['e_ms_age_hi']),
                 yerr=(result_garces2011['ms_age_err_low']/1e9,result_garces2011['ms_age_err_high']/1e9),fmt='.')
    x = np.linspace(0,4)
    ax4.plot(x,x,'-',color=color_line,linewidth=lw)
    ax4.set_xlabel('MS Age')
    ax4.tick_params('both',direction='in',top=True,right=True)
    ax4.tick_params('y',which='minor',direction='in',right=True)
    
    f.text(0.5, 0.01, 'Garces 2011 ', ha='center')
    f.text(0.00, 0.5, 'This work', va='center', rotation='vertical')
    
    plt.tight_layout()
    plt.show()
    
#def plot_distributions(i,results,final_mass_compare,intial_mass_compare,cooling_age_compare,
#                       ms_age_compare,total_age_compare,xscale='log'):
'''