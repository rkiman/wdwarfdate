#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
#import os
from .bayesian_age import ln_posterior_prob#, ln_prior
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
        #Reset sampler and remove the save likelihood evaluation that were saved
        #while calculating autocorrelation time and burn in
        #save_likelihoods_file = wd_path_id +'.txt'
        #os.remove(save_likelihoods_file)
        
        #Run mcmc to calculate parameters
        #p,_,_ = sampler.run_mcmc(p, n_indep_samples*autoc_time);
        
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
    ax1.set_ylabel(r'$\log_{10}($MS Age$/yr)$')
    
    for i in range(50):
        ax2.plot(chain[i,:,1],color='k',alpha=0.05)
        ax2.axhline(y=np.median(flat_samples[:,1]),color='k')
    ax2.set_ylabel(r'$\log_{10}($Cooling Age$/yr)$')
    
    for i in range(50):
        ax3.plot(chain[i,:,2],color='k',alpha=0.05)
        ax3.axhline(y=np.median(flat_samples[:,2]),color='k')
    ax3.set_ylabel(r'$delta_m$')
    plt.tight_layout()
    plt.savefig(wd_path_id + '_walkers.png')
    plt.close(f)
    
    labels=[r'$\log_{10}($msa$/yr)$',r'$\log_{10}($ca$/yr)$',r'$delta_m$']
    
    fig = corner.corner(flat_samples, labels=labels, 
                        quantiles=[.16,.50,.84], 
                        show_titles=True, title_kwargs={"fontsize": 12})
    fig.savefig(wd_path_id + '_corner_plot.png',dpi=300)
    plt.close(fig)
    
    
def kl_divergence(p, q):
    mask = (q != 0.0) * (p != 0.0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

    
def get_initial_conditions(teff0,logg0,model_wd,model_ifmr,
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
    
    return init_params
    '''
    Creates a grid of main sequence age and cooling age and evaluates the
    posterior on that grid. Also from the grid gets the distribution of 
    posterior for the two parameters: main sequence age and cooling age.
    
    #Set up grid
    Ntot = n*n
    x = np.ones(Ntot)*np.nan
    y = np.ones(Ntot)*np.nan

    x0 = np.linspace(4,13,n)
    y0 = np.linspace(4,13,n)

    for ii in range(n):
        x[ii*n:(ii+1)*n] = np.ones(n)*x0[ii]
        y[ii*n:(ii+1)*n] = y0
    
    #Evaluate posterior on grid    
    post_list = []
    for x_i,y_i in zip(x,y):
        post = ln_posterior_prob([x_i,y_i,0], #set delta_m = 0 
                                 teff0,e_teff0,logg0,e_logg0,models0)
        post_list.append(post)
    post_list = np.array(post_list)

    #Colapse grid to obtain distributions of posterior in main sequence age and
    #cooling age dimensions
    post_ms_age = np.array([sum(np.exp(post_list[x==x0_i])) for x0_i in x0])
    post_cooling_age = np.array([sum(np.exp(post_list[y==y0_i])) for y0_i in y0])
    
    #Normalize distribution so the sum is 1
    norm_post_ms = sum(post_ms_age)*(x0[1]-x0[0])
    norm_post_cooling = sum(post_cooling_age)*(y0[1]-y0[0])
    post_ms_age = post_ms_age/norm_post_ms
    post_cooling_age = post_cooling_age/norm_post_cooling

    axis = x0, y0, x, y
    
    idx = np.argmax(post_list)
    init_max_like = np.array([x[idx],y[idx],0])
    
    return axis, post_list, post_ms_age, post_cooling_age, init_max_like
    '''