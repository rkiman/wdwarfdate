import numpy as np
from coolingage import calc_cooling_age
from final2initial_mass import calc_initial_mass
from ms_age import calc_ms_age
import matplotlib.pyplot as plt
from astropy.table import Table

def calc_wd_age(teff,e_teff,logg,e_logg,n_mc=2000,
                model_wd='DA',feh='p0.00',vvcrit='0.0',
                model_ifmr = 'Cummings 2018',
                return_distributions=False):
    
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