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
        teff_dist.append(np.random.normal(teff[i],e_teff[i],n_mc))
        logg_dist.append(np.random.normal(logg[i],e_logg[i],n_mc))
    teff_dist,logg_dist = np.array(teff_dist),np.array(logg_dist)
        
    cooling_age_dist,final_mass_dist = calc_cooling_age(teff_dist,logg_dist,
                                                        n_mc,N,model=model_wd)
    
    initial_mass_dist = calc_initial_mass(model_ifmr,final_mass_dist,n_mc)
    
    ms_age_dist = calc_ms_age(initial_mass_dist,feh=feh,vvcrit=vvcrit)
    
    total_age_dist = cooling_age_dist + ms_age_dist
    
    results = Table()
    
    results['final_mass_median'] = np.array([np.nanpercentile(x,50) for x in final_mass_dist])
    results['final_mass_err_high'] = np.array([np.nanpercentile(x,84.1345) for x in final_mass_dist])
    results['final_mass_err_low'] = np.array([np.nanpercentile(x,15.8655) for x in final_mass_dist])
    
    results['initial_mass_median'] = np.array([np.nanpercentile(x,50) for x in initial_mass_dist])
    results['initial_mass_err_high'] = np.array([np.nanpercentile(x,84.1345) for x in initial_mass_dist])
    results['initial_mass_err_low'] = np.array([np.nanpercentile(x,15.8655) for x in initial_mass_dist])

    results['cooling_age_median'] = np.array([np.nanpercentile(x,50) for x in cooling_age_dist])
    results['cooling_age_err_high'] = np.array([np.nanpercentile(x,84.1345) for x in cooling_age_dist])
    results['cooling_age_err_low'] = np.array([np.nanpercentile(x,15.8655) for x in cooling_age_dist])

    results['ms_age_median'] = np.array([np.nanpercentile(x,50) for x in ms_age_dist])
    results['ms_age_err_high'] = np.array([np.nanpercentile(x,84.1345) for x in ms_age_dist])
    results['ms_age_err_low'] = np.array([np.nanpercentile(x,15.8655) for x in ms_age_dist])

    results['total_age_median'] = np.array([np.nanpercentile(x,50) for x in total_age_dist])
    results['total_age_err_high'] = np.array([np.nanpercentile(x,84.1345) for x in total_age_dist])
    results['total_age_err_low'] = np.array([np.nanpercentile(x,15.8655) for x in total_age_dist])    
    
    if(return_distributions):
        results['final_mass_dist'] = final_mass_dist
        results['initial_mass_dist'] = initial_mass_dist
        results['cooling_age_dist'] = cooling_age_dist
        results['ms_age_dist'] = ms_age_dist
        results['total_age_dist'] = total_age_dist
    return results


'''
wd = Table.read('/Users/rociokiman/Documents/M-dwarfs-Age-Activity-Relation/Catalogs/WD_M_Pairs_new.csv')
'''
logg = np.array([8.195693])
e_logg = np.array([0.160218])

teff = np.array([14964.260506])
e_teff = np.array([1375.595732])

results = calc_wd_age(teff,e_teff,logg,e_logg,n_mc=2000,
            model_wd='DA',feh='p0.00',vvcrit='0.4',return_distributions=True)

#print('Cooling Age: {}yr +/- {}yr\nMS Age: {}yr +\- {}yr\nTotal Age: {}yr +\- {}yr'.format(results['cooling_age_median'][0]/1e6,
#      results['cooling_age_std'][0]/1e6,results['ms_age_median'][0]/1e6,results['ms_age_std'][0]/1e6,
#      results['total_age_median'][0]/1e6,results['total_age_std'][0]/1e6))

plt.hist(results['final_mass_dist'][0],bins=20)
plt.axvline(x=results['final_mass_median'][0],color='k',linestyle='--')
plt.axvline(x=results['final_mass_err_high'][0],color='k',linestyle='--')
plt.axvline(x=results['final_mass_err_low'][0],color='k',linestyle='--')
#plt.axvline(x=650,color='k')
plt.show()

plt.hist(results['initial_mass_dist'][0],bins=20)
plt.axvline(x=results['initial_mass_median'][0],color='k',linestyle='--')
plt.axvline(x=results['initial_mass_err_high'][0],color='k',linestyle='--')
plt.axvline(x=results['initial_mass_err_low'][0],color='k',linestyle='--')
#plt.axvline(x=650,color='k')
plt.show()

plt.hist(results['ms_age_dist'][0]/1e6,bins=np.linspace(0,2000,30))
plt.axvline(x=results['ms_age_median'][0]/1e6,color='k',linestyle='--')
plt.axvline(x=results['ms_age_err_high'][0]/1e6,color='k',linestyle='--')
plt.axvline(x=results['ms_age_err_low'][0]/1e6,color='k',linestyle='--')
#plt.axvline(x=650,color='k')
plt.show()

plt.hist(results['total_age_dist'][0]/1e6,bins=np.linspace(0,2000,30))
plt.axvline(x=results['total_age_median'][0]/1e6,color='k',linestyle='--')
plt.axvline(x=results['total_age_err_high'][0]/1e6,color='k',linestyle='--')
plt.axvline(x=results['total_age_err_low'][0]/1e6,color='k',linestyle='--')
plt.axvline(x=650,color='k')
plt.show()

'''
n_mc = 2000

plt.plot(final_mass,final_mass_median,'.')
plt.xlabel('Final Mass')
plt.show()

plt.plot(initial_mass_median,final_mass,'.')
plt.xlabel('initial mass')
plt.ylabel('final mass')
plt.show()

plt.plot(wd['WD_CoolingAge (yr)'],cooling_age_median,'.')
plt.xlabel('Cooling Age')
plt.show()

x = np.linspace(10**8,10**10,10)
plt.loglog(wd['MS_Age (yr)'],ms_age_median,'.')
plt.loglog(x,x,'-k')
plt.xlabel('MS Age')
plt.show()

plt.loglog(wd['Total_Age(yr)'],total_age_median,'.')
plt.loglog(x,x,'-k')
plt.xlabel('Total Age')
plt.show()
'''