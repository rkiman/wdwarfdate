#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from .bayesian_age import ln_posterior_prob
from .check_convergence import calc_auto_corr_time
from .cooling_age import calc_cooling_age
from .ms_age import calc_ms_age
from .ifmr import calc_initial_mass
    
def run_mcmc(teff0, e_teff0, logg0, e_logg0, models0, 
             init_params, comparison,
             nburn_in,max_n,n_indep_samples,plot):
    '''
    Starting from the maximum likelihood ages (main sequence age and cooling
    age), samples the posterior to get the likelihood evaluations of the rest 
    of the parameters (final mass, initial mass and total age)
    models0 : list. [model_ifmr,isochrone_model,cooling_models,wd_path_id]
    '''

    _,_,_,wd_path_id = models0
    ndim, nwalkers = 3, 70 #nwalkers > 2*ndim
    
    #Initialize walkers    
    p0 = np.array([init_params
                   + np.random.uniform(-.05,.05,3) for i in range(nwalkers)])
    
    #Remove the name of the file so it doesn't save likelihood evaluations
    models0[3] = ''
    #Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_posterior_prob,
                                    args=[teff0,e_teff0,logg0,e_logg0,models0])
    
    #Running burn in
    p0_new,_,_ = sampler.run_mcmc(p0, nburn_in)
    
    save_likelihoods_file = wd_path_id +'.txt'
    with open(save_likelihoods_file,'a') as f:
        models0[3] = f
        #Initialize sampler again but now so it saves likelihood evaluations
        sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_posterior_prob,
                                        args=[teff0,e_teff0,logg0,e_logg0,
                                              models0])
        n_steps = int(max_n/100)
        
        index = 0
        autocorr = np.empty(max_n)
        
        # This will be useful to testing convergence
        old_tau = np.inf
        
        # going to run the mcmc in groups of 100 steps
        for x in range(n_steps):
            p0_new,_,_ = sampler.run_mcmc(p0_new, 100)
            chain = sampler.chain
            # Compute the autocorrelation time so far
            tau = calc_auto_corr_time(chain)
            autocorr[index] = np.mean(tau)
            index += 1
        
            # Check convergence
            converged = np.all(tau * n_indep_samples < (x+1)*100)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        
        #Plot convergence
        if(plot==True):
            N = 100 * np.arange(1, index + 1)
            plt.plot(N, N / 100.0, "--k", label=r"$\tau = N/100$")
            plt.loglog(N, autocorr[:index], "-")
            plt.xlabel("number of samples, $N$")
            plt.ylabel(r"mean $\hat{\tau}$")
            plt.legend(fontsize=14)
            plt.grid()
            plt.savefig(wd_path_id+'_corr_time.png')
            plt.close()
        
        #Obtain chain of samples
        chain = sampler.chain[:,:,:]
        flat_samples = chain.reshape((-1,ndim))
        if(plot==True):
            plot_results_mcmc(chain,ndim,wd_path_id)
            
    return flat_samples

def plot_results_mcmc(chain,ndim,wd_path_id):
    flat_samples = chain.reshape((-1,ndim))
    
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(8,3))
    for i in range(50):
        ax1.plot(chain[i,:,0],color='k',alpha=0.05)
        ax1.axhline(y=np.median(flat_samples[:,0]),color='k')
    ax1.set_ylabel(r'$\log_{10}(t_{\rm ms}/{\rm yr})$')
    
    for i in range(50):
        ax2.plot(chain[i,:,1],color='k',alpha=0.05)
        ax2.axhline(y=np.median(flat_samples[:,1]),color='k')
    ax2.set_ylabel(r'$\log_{10}(t_{\rm cool}/{\rm yr})$')
    
    for i in range(50):
        ax3.plot(chain[i,:,2],color='k',alpha=0.05)
        ax3.axhline(y=np.median(flat_samples[:,2]),color='k')
    ax3.set_ylabel(r'$\Delta_{\rm m}$')
    plt.tight_layout()
    plt.savefig(wd_path_id + '_walkers.png')
    plt.close(f)
    
    labels=[r'$\log_{10}(t_{\rm ms}/{\rm yr})$',
            r'$\log_{10}(t_{\rm cool}/{\rm yr})$',
            r'$\Delta _{\rm m}$']
    
    fig = corner.corner(flat_samples, labels=labels, 
                        quantiles=[.16,.50,.84], 
                        show_titles=True, title_kwargs={"fontsize": 12})
    fig.savefig(wd_path_id + '_corner_plot.png',dpi=300)
    plt.close(fig)

    
def get_initial_conditions(teff0,e_teff0,logg0,e_logg0,model_wd,model_ifmr,
                           feh,vvcrit):
    teff_dist = np.array([[teff0]])
    logg_dist = np.array([[logg0]])
    cool_age_dist,final_mass_dist = calc_cooling_age(teff_dist,logg_dist,
                                                     1,model_wd)
    initial_mass_dist = calc_initial_mass(model_ifmr,final_mass_dist)
    ms_age_dist = calc_ms_age(initial_mass_dist,feh,vvcrit)
    
    init_params = np.array([np.log10(ms_age_dist[0][0]),
                            np.log10(cool_age_dist[0][0]),
                            0])

    if(any(np.isnan(init_params))):
        teff_dist = np.array([np.random.normal(teff0,e_teff0,1000)])
        logg_dist = np.array([np.random.normal(logg0,e_logg0,1000)])
        cool_age_dist,final_mass_dist = calc_cooling_age(teff_dist,logg_dist,
                                                         1,model_wd)
        initial_mass_dist = calc_initial_mass(model_ifmr,final_mass_dist)
        ms_age_dist = calc_ms_age(initial_mass_dist,feh,vvcrit)
        
        init_params = np.array([np.nanmedian(np.log10(ms_age_dist[0])),
                                np.nanmedian(np.log10(cool_age_dist[0])),
                                0])

    return init_params