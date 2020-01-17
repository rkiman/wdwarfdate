import numpy as np
from coolingage import calc_cooling_age
from final2initial_mass import calc_initial_mass
from ms_age import calc_ms_age
#import matplotlib.pyplot as plt
from astropy.table import Table

def calc_wd_age(teff,e_teff,logg,e_logg,n_mc=2000,
                model_wd='DA',feh='p0.00',vvcrit='0.4'):
    
    cooling_age,final_mass = calc_cooling_age(teff,e_teff,logg,e_logg,n_mc,
                                               model=model_wd)
    
    initial_mass = calc_initial_mass(final_mass,n_mc)
    
    ms_age = calc_ms_age(initial_mass,feh=feh,vvcrit=vvcrit)
    
    results = Table()
    
    results['final_mass_median'] = np.array([np.nanmedian(x) for x in final_mass])
    results['final_mass_std'] = np.array([np.nanstd(x) for x in final_mass])
    results['initial_mass_median'] = np.array([np.nanmedian(x) for x in initial_mass])
    results['initial_mass_std'] = np.array([np.nanstd(x) for x in initial_mass])

    results['cooling_age_median'] = np.array([np.nanmedian(x) for x in cooling_age])
    results['cooling_age_std'] = np.array([np.nanstd(x) for x in cooling_age])

    results['ms_age_median'] = np.array([np.nanmedian(x) for x in ms_age])
    results['ms_age_std'] = np.array([np.nanstd(x) for x in ms_age])

    results['total_age_median'] = np.array([np.nanmedian(x+y) for x,y in zip(ms_age,cooling_age)])
    results['total_age_std'] = np.array([np.nanstd(x+y) for x,y in zip(ms_age,cooling_age)])    
    
    return results


'''
wd = Table.read('/Users/rociokiman/Documents/M-dwarfs-Age-Activity-Relation/Catalogs/WD_M_Pairs_new.csv')

logg = np.array([9.20])
e_logg = np.array([0.07])

teff = np.array([42700])
e_teff = np.array([800])

results = calc_wd_age(teff,e_teff,logg,e_logg,n_mc=2000,
            model_wd='DA',feh='p0.00',vvcrit='0.4')

print(results)
print('Cooling Age: {}yr +/- {}yr\nMS Age: {}yr +\- {}yr\nTotal Age: {}yr +\- {}yr'.format(results['cooling_age_median'][0]/1e6,
      results['cooling_age_std'][0]/1e6,results['ms_age_median'][0]/1e6,results['ms_age_std'][0]/1e6,
      results['total_age_median'][0]/1e6,results['total_age_std'][0]/1e6))

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