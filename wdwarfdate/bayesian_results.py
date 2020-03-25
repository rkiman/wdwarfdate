#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .bayesian_age import ln_posterior_prob, ln_prior

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_prior_dist(teff0, e_teff0, logg0, e_logg0, models0, n=500):
    Ntot = n*n
    x = np.ones(Ntot)*np.nan
    y = np.ones(Ntot)*np.nan

    x0 = np.linspace(4,11,n)
    y0 = np.linspace(4,11,n)

    for ii in range(n):
        x[ii*n:(ii+1)*n] = np.ones(n)*x0[ii]
        y[ii*n:(ii+1)*n] = y0
    
    prior_list = []
    for x_i,y_i in zip(x,y):
        prior = ln_prior([x_i,y_i],teff0,e_teff0,logg0,e_logg0,models0)
        prior_list.append(prior)
    prior_list = np.array(prior_list)
    
    prior_ms_age = np.array([sum(np.exp(prior_list[x==x0_i])) for x0_i in x0])
    prior_cooling_age = np.array([sum(np.exp(prior_list[y==y0_i])) for y0_i in y0])
    norm_x = sum(prior_ms_age)*(x0[1]-x0[0])
    norm_y = sum(prior_cooling_age)*(y0[1]-y0[0])
    
    prior_ms_age = prior_ms_age/norm_x
    prior_cooling_age = prior_cooling_age/norm_y
    
    axis = x0, y0, x, y
        
    return axis, prior_list, prior_ms_age, prior_cooling_age
    
def get_post_dist(teff0, e_teff0, logg0, e_logg0, models0, n=500):
    Ntot = n*n
    x = np.ones(Ntot)*np.nan
    y = np.ones(Ntot)*np.nan

    x0 = np.linspace(4,11,n)
    y0 = np.linspace(4,11,n)

    for ii in range(n):
        x[ii*n:(ii+1)*n] = np.ones(n)*x0[ii]
        y[ii*n:(ii+1)*n] = y0
        
    post_list = []
    for x_i,y_i in zip(x,y):
        post = ln_posterior_prob([x_i,y_i],teff0,e_teff0,logg0,e_logg0,models0)
        post_list.append(post)
    post_list = np.array(post_list)
    
    post_ms_age = np.array([sum(np.exp(post_list[x==x0_i])) for x0_i in x0])
    post_cooling_age = np.array([sum(np.exp(post_list[y==y0_i])) for y0_i in y0])
    norm_post_ms = sum(post_ms_age)*(x0[1]-x0[0])
    norm_post_cooling = sum(post_cooling_age)*(y0[1]-y0[0])
    
    post_ms_age = post_ms_age/norm_post_ms
    post_cooling_age = post_cooling_age/norm_post_cooling

    axis = x0, y0, x, y
    
    return axis, post_list, post_ms_age, post_cooling_age

def plot_prior_post_dist(teff0, e_teff0, logg0, e_logg0, models0, n=500, 
                         truths = []):
    
    axis_prior, prior_list, prior_ms_age, prior_cooling_age = get_prior_dist(teff0, e_teff0, logg0, e_logg0, models0, n)
    axis_post, post_list, post_ms_age, post_cooling_age = get_post_dist(teff0, e_teff0, logg0, e_logg0, models0, n)
    
    x0_prior, y0_prior, x_prior, y_prior = axis_prior
    x0_post, y0_post, x_post, y_post = axis_post
    
    idx = np.argmax(post_list)
    init_max_like = np.array([x_post[idx],y_post[idx]])
    
    plt.scatter(x_prior,y_prior,c=np.exp(prior_list),s=5)
    if(len(truths)!=0):
        plt.axvline(x=truths[0])
        plt.axhline(y=truths[1])
    plt.colorbar(label='Posterior')
    plt.xlabel('log(ms_age)')
    plt.ylabel('log(cooling_age)')
    plt.show()
    
    plt.scatter(x_post,y_post,c=np.exp(post_list),s=5)
    if(len(truths)!=0):
        plt.axvline(x=truths[0])
        plt.axhline(y=truths[1])
    plt.colorbar(label='Posterior')
    plt.xlabel('log(ms_age)')
    plt.ylabel('log(cooling_age)')
    plt.show()

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.plot(x0_prior,prior_ms_age,label='Prior')
    ax1.plot(x0_post,post_ms_age,label='Posterior')
    ax1.set_ylim(0)
    if(all(x0_prior == x0_post)):
        ax1.legend(title='KL(Post||Prior) = {0:.2f}'.format(kl_divergence(post_ms_age, prior_ms_age)))
    ax1.set_xlabel(r'$\log_{10}($MS Age$/yr)$')
    
    ax2.plot(x0_prior,prior_cooling_age,label='Prior')
    ax2.plot(x0_post,post_cooling_age,label='Posterior')
    ax2.set_ylim(0)
    if(all(x0_prior == x0_post)):
        ax2.legend(title='KL(Post||Prior) = {0:.2f}'.format(kl_divergence(post_cooling_age, prior_cooling_age)))
    ax2.set_xlabel(r'$\log_{10}($Cooling Age$/yr)$')
    plt.show()
    
    return init_max_like