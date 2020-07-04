#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def calc_percentiles(ln_ms_age, ln_cooling_age, ln_total_age, initial_mass, 
                     final_mass, high_perc, low_perc, datatype='log'):

    if(datatype=='log'):
        ms_age_median = np.nanpercentile(ln_ms_age,50)
        ms_age_err_low = ms_age_median - np.nanpercentile(ln_ms_age,low_perc)
        ms_age_err_high = np.nanpercentile(ln_ms_age,high_perc) - ms_age_median
        
        cooling_age_median = np.nanpercentile(ln_cooling_age,50)
        cooling_age_err_low = cooling_age_median - np.nanpercentile(ln_cooling_age,low_perc)
        cooling_age_err_high = np.nanpercentile(ln_cooling_age,high_perc) - cooling_age_median
        
        total_age_median = np.nanpercentile(ln_total_age,50)
        total_age_err_low = total_age_median - np.nanpercentile(ln_total_age,low_perc)
        total_age_err_high = np.nanpercentile(ln_total_age,high_perc) - total_age_median
        
        initial_mass_median = np.nanpercentile(initial_mass,50)
        initial_mass_err_low = initial_mass_median - np.nanpercentile(initial_mass,low_perc)
        initial_mass_err_high = np.nanpercentile(initial_mass,high_perc) - initial_mass_median
        
        final_mass_median = np.nanpercentile(final_mass,50)
        final_mass_low = final_mass_median - np.nanpercentile(final_mass,low_perc)
        final_mass_high = np.nanpercentile(final_mass,high_perc) - final_mass_median
        
    elif(datatype=='Gyr'):
        ms_age_median = np.nanpercentile((10**ln_ms_age)/1e9,50)
        ms_age_err_low = ms_age_median - np.nanpercentile((10**ln_ms_age)/1e9,low_perc)
        ms_age_err_high = np.nanpercentile((10**ln_ms_age)/1e9,high_perc) - ms_age_median
        
        cooling_age_median = np.nanpercentile((10**ln_cooling_age)/1e9,50)
        cooling_age_err_low = cooling_age_median - np.nanpercentile((10**ln_cooling_age)/1e9,low_perc)
        cooling_age_err_high = np.nanpercentile((10**ln_cooling_age)/1e9,high_perc) - cooling_age_median
        
        total_age_median = np.nanpercentile((10**ln_total_age)/1e9,50)
        total_age_err_low = total_age_median - np.nanpercentile((10**ln_total_age)/1e9,low_perc)
        total_age_err_high = np.nanpercentile((10**ln_total_age)/1e9,high_perc) - total_age_median
        
        initial_mass_median = np.nanpercentile(initial_mass,50)
        initial_mass_err_low = initial_mass_median - np.nanpercentile(initial_mass,low_perc)
        initial_mass_err_high = np.nanpercentile(initial_mass,high_perc) - initial_mass_median
        
        final_mass_median = np.nanpercentile(final_mass,50)
        final_mass_low = final_mass_median - np.nanpercentile(final_mass,low_perc)
        final_mass_high = np.nanpercentile(final_mass,high_perc) - final_mass_median
        
    elif(datatype=='yr'):
        ms_age_median = np.nanpercentile(10**ln_ms_age,50)
        ms_age_err_low = ms_age_median - np.nanpercentile(10**ln_ms_age,low_perc)
        ms_age_err_high = np.nanpercentile(10**ln_ms_age,high_perc) - ms_age_median
        
        cooling_age_median = np.nanpercentile(10**ln_cooling_age,50)
        cooling_age_err_low = cooling_age_median - np.nanpercentile(10**ln_cooling_age,low_perc)
        cooling_age_err_high = np.nanpercentile(10**ln_cooling_age,high_perc) - cooling_age_median
        
        total_age_median = np.nanpercentile(10**ln_total_age,50)
        total_age_err_low = total_age_median - np.nanpercentile(10**ln_total_age,low_perc)
        total_age_err_high = np.nanpercentile(10**ln_total_age,high_perc) - total_age_median
        
        initial_mass_median = np.nanpercentile(initial_mass,50)
        initial_mass_err_low = initial_mass_median - np.nanpercentile(initial_mass,low_perc)
        initial_mass_err_high = np.nanpercentile(initial_mass,high_perc) - initial_mass_median
        
        final_mass_median = np.nanpercentile(final_mass,50)
        final_mass_low = final_mass_median - np.nanpercentile(final_mass,low_perc)
        final_mass_high = np.nanpercentile(final_mass,high_perc) - final_mass_median
        
    return [ms_age_median,ms_age_err_low,ms_age_err_high,
            cooling_age_median,cooling_age_err_low,cooling_age_err_high,
            total_age_median,total_age_err_low,total_age_err_high,
            initial_mass_median,initial_mass_err_low,initial_mass_err_high,
            final_mass_median,final_mass_low,final_mass_high]

def plot_distributions(teff0, logg0, ln_ms_age, ln_cooling_age, ln_total_age, 
                       initial_mass, final_mass, high_perc, low_perc, 
                       comparison=[], name = 'run'):
    
    results = calc_percentiles(ln_ms_age, ln_cooling_age, ln_total_age, initial_mass, 
                               final_mass, high_perc, low_perc, datatype='log')
    title = r"${{{0:.2f}}}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
    f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize=(12,3))
    ax1.hist(ln_ms_age,bins=20)
    ax1.axvline(x=results[0],color='k')
    ax1.axvline(x=results[0]-results[1],color='k',linestyle='--')
    ax1.axvline(x=results[0]+results[2],color='k',linestyle='--')
    if(len(comparison) != 0):
        ax1.axvline(x=comparison[0],color='r')
    ax1.set_xlabel(r'$\log _{10}($MS Age$/{\rm yr})$')
    ax1.set_title(title.format(results[0],results[1],results[2]))
    
    ax2.hist(ln_cooling_age,bins=20)
    ax2.axvline(x=results[3],color='k')
    ax2.axvline(x=results[3]-results[4],color='k',linestyle='--')
    ax2.axvline(x=results[3]+results[5],color='k',linestyle='--')
    if(len(comparison) != 0):
        ax2.axvline(x=comparison[1],color='r')
    ax2.set_xlabel(r'$\log _{10}($Cooling Age$/{\rm yr})$')
    ax2.set_title(title.format(results[3],results[4],results[5]))
    
    ax3.hist(ln_total_age,bins=20)
    ax3.axvline(x=results[6],color='k')
    ax3.axvline(x=results[6]-results[7],color='k',linestyle='--')
    ax3.axvline(x=results[6]+results[8],color='k',linestyle='--')
    if(len(comparison) != 0):
        ax3.axvline(x=comparison[2],color='r')
    ax3.set_xlabel(r'$\log _{10}($Total$/{\rm yr})$')
    ax3.set_title(title.format(results[6],results[7],results[8]))
    
    ax4.hist(initial_mass,bins=20)
    ax4.axvline(x=results[9],color='k')
    ax4.axvline(x=results[9]-results[10],color='k',linestyle='--')
    ax4.axvline(x=results[9]+results[11],color='k',linestyle='--')
    if(len(comparison) != 0):
        ax4.axvline(x=comparison[3],color='r')
    ax4.set_xlabel('Initial Mass')
    ax4.set_title(title.format(results[9],results[10],results[11]))
    
    ax5.hist(final_mass,bins=20)
    ax5.axvline(x=results[12],color='k')
    ax5.axvline(x=results[12]-results[13],color='k',linestyle='--')
    ax5.axvline(x=results[12]+results[14],color='k',linestyle='--')
    if(len(comparison) != 0):
        ax5.axvline(x=comparison[4],color='r')
    ax5.set_xlabel('Final Mass')
    ax5.set_title(title.format(results[12],results[13],results[14]))
    
    plt.tight_layout()
    plt.savefig(name + '_distributions.png',dpi=300)
    plt.close(f)