#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
from .bayesian_age import ln_posterior_prob, ln_prior
from .check_convergence import calc_auto_corr_time

def kl_divergence(p, q):
    mask = (q != 0.0) * (p != 0.0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

    
def get_post_dist(teff0, e_teff0, logg0, e_logg0, models0, n=100):
    '''
    Creates a grid of main sequence age and cooling age and evaluates the
    posterior on that grid. Also from the grid gets the distribution of 
    posterior for the two parameters: main sequence age and cooling age.
    '''
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
    
def run_mcmc(teff0, e_teff0, logg0, e_logg0, models0, 
             init_params, comparison,plot,
             nburn_in,n_calc_auto_corr,n_indep_samples,
             auto_corr_time):
    '''
    Starting from the maximum likelihood ages (main sequence age and cooling
    age), samples the posterior to get the likelihood evaluations of the rest 
    of the parameters (final mass, initial mass and total age)
    '''

    _,_,_,wd_path_id = models0
    ndim, nwalkers = 3, 70 #nwalkers > 2*ndim
    
    #Obtain the maximum likelihood parameters if they are not given.
    #if(len(init_params)==0 and plot==False):
    _,_,_,_,init_max_like = get_post_dist(teff0, e_teff0, logg0, e_logg0,
                                          models0, n=500)
    init_params = init_max_like 
        
    #Initialize walkers    
    p0 = np.array([init_params
                   + np.random.uniform(-.05,.05,3) for i in range(nwalkers)])
    
    #Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_posterior_prob,
                                    args=[teff0,e_teff0,logg0,e_logg0,models0])
    
    #Running burn in
    p0_new,_,_ = sampler.run_mcmc(p0, nburn_in)
    sampler.reset()
    
    if(auto_corr_time==0):
        #Running mcmc to calculate auto-correlation time'
        p,_,_ = sampler.run_mcmc(p0_new, n_calc_auto_corr)
        chain = sampler.chain
        autoc_time = calc_auto_corr_time(chain,plot,
                                         wd_path_id+'_corr_time.png')
        sampler.reset()
    else:
        autoc_time=auto_corr_time
        p=p0_new
    
    #Reset sampler and remove the save likelihood evaluation that were saved
    #while calculating autocorrelation time and burn in
    save_likelihoods_file = wd_path_id +'.txt'
    os.remove(save_likelihoods_file)
    
    #Run mcmc to calculate parameters
    p,_,_ = sampler.run_mcmc(p, n_indep_samples*autoc_time);
    
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


def plot_prior_post_dist(teff0, e_teff0, logg0, e_logg0, models0, n=500, 
                         comparison = [], figname = 'run'):
    '''
    Makes plots with the distributions of the prior and posterior for the
    main sequence age and the cooling age (the two independent parameters in
    the model). 
    '''
    
    #Obtain the distributions of prior and posterior for main sequence age
    #and cooling age
    r_prior = get_prior_dist(teff0, e_teff0, logg0, e_logg0, models0, n)
    r_post = get_post_dist(teff0, e_teff0, logg0, e_logg0, models0, n)
    
    axis_prior, prior_list, prior_ms_age, prior_cooling_age = r_prior
    axis_post, post_list, post_ms_age, post_cooling_age, init_max_like = r_post
    
    x0_prior, y0_prior, x_prior, y_prior = axis_prior
    x0_post, y0_post, x_post, y_post = axis_post
    
    #Plot parameter space for prior
    f,((ax3,ax4),(ax1,ax2)) = plt.subplots(2,2,figsize=(10,7))
    ax3.scatter(x_prior,y_prior,c=np.exp(prior_list),
                s=5,cmap='Greys',label='Prior')
    if(len(comparison)!=0):
        ax3.axvline(x=comparison[0])
        ax3.axhline(y=comparison[1])
    ax3.set_xlabel(r'$\log_{10}($MS Age$/yr)$')
    ax3.set_ylabel(r'$\log_{10}($Cooling Age$/yr)$')
    ax3.legend()
    
    #Plot parameter space for posterior
    ax4.scatter(x_post,y_post,c=np.exp(post_list),
                s=5,cmap='Greys',label='Posterior')
    if(len(comparison)!=0):
        ax4.axvline(x=comparison[0])
        ax4.axhline(y=comparison[1])
    ax4.set_xlim(np.nanmin(x_post[np.exp(post_list)!=0])-0.1,
                 np.nanmax(x_post[np.exp(post_list)!=0])+0.1)
    ax4.set_ylim(np.nanmin(y_post[np.exp(post_list)!=0])-0.1,
                 np.nanmax(y_post[np.exp(post_list)!=0])+0.1)
    ax4.set_xlabel(r'$\log_{10}($MS Age$/yr)$')
    ax4.set_ylabel(r'$\log_{10}($Cooling Age$/yr)$')
    ax4.legend()
    
    #Plot distribution of prior and posterior for main sequence age
    ax1.plot(x0_prior,prior_ms_age,label='Prior')
    ax1.plot(x0_post,post_ms_age,label='Posterior')
    ax1.set_ylim(0)
    if(all(x0_prior == x0_post)):
        kl_div = kl_divergence(post_ms_age, prior_ms_age)
    else:
        print("Can't calculate KL divergence because the x axis are different")
        ax1.legend(title='KL(Post||Prior) = {0:.2f}'.format(kl_div))
    ax1.set_xlabel(r'$\log_{10}($MS Age$/yr)$')
    
    #Plot distribution of prior and posterior for cooling age
    ax2.plot(x0_prior,prior_cooling_age,label='Prior')
    ax2.plot(x0_post,post_cooling_age,label='Posterior')
    ax2.set_ylim(0)
    if(all(x0_prior == x0_post)):
        kl_div = kl_divergence(post_cooling_age, prior_cooling_age)
        ax2.legend(title='KL(Post||Prior) = {0:.2f}'.format(kl_div))
    else:
        print("Can't calculate KL divergence because the x axis are different")
    ax2.set_xlabel(r'$\log_{10}($Cooling Age$/yr)$')
    plt.tight_layout()
    plt.savefig(figname + '_post_prior_dist.png',dpi=300)
    plt.close(f)
    
    #Return the maximum likelihood [ms age,cooling age]
    return init_max_like

def get_prior_dist(teff0, e_teff0, logg0, e_logg0, models0, n=500):
    '''
    Creates a grid of main sequence age and cooling age and evaluates the
    prior on that grid. Also from the grid gets the distribution of prior for
    the two parameters: main sequence age and cooling age.
    '''
    
    #Set up grid
    Ntot = n*n
    x = np.ones(Ntot)*np.nan
    y = np.ones(Ntot)*np.nan

    x0 = np.linspace(4,13,n)
    y0 = np.linspace(4,13,n)

    for ii in range(n):
        x[ii*n:(ii+1)*n] = np.ones(n)*x0[ii]
        y[ii*n:(ii+1)*n] = y0
    
    #Evaluate prior on grid
    prior_list = []
    for x_i,y_i in zip(x,y):
        prior = ln_prior([x_i,y_i],teff0,e_teff0,logg0,e_logg0,models0)
        prior_list.append(prior)
    prior_list = np.array(prior_list)
    
    #Colapse grid to obtain distributions of prior in main sequence age and
    #cooling age dimensions
    prior_ms_age = np.array([sum(np.exp(prior_list[x==x0_i])) for x0_i in x0])
    prior_cooling_age = np.array([sum(np.exp(prior_list[y==y0_i])) for y0_i in y0])
    
    #Normalize distribution so the sum is 1
    norm_x = sum(prior_ms_age)*(x0[1]-x0[0])
    norm_y = sum(prior_cooling_age)*(y0[1]-y0[0])   
    prior_ms_age = prior_ms_age/norm_x
    prior_cooling_age = prior_cooling_age/norm_y
    
    axis = x0, y0, x, y
        
    return axis, prior_list, prior_ms_age, prior_cooling_age
