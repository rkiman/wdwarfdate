import numpy as np
from coolingage import calc_cooling_age
from final2initial_mass import calc_initial_mass
from ms_age import calc_ms_age
#import matplotlib.pyplot as plt
#from astropy.table import Table

def calc_wd_age(teff,e_teff,logg,e_logg,final_mass,e_final_mass,n_mc=2000,
                model_wd='DA',feh='p0.00',vvcrit='0.4'):

    N = len(logg)
    
    cooling_age,final_mass2 = calc_cooling_age(teff,e_teff,logg,e_logg,n_mc,
                                               model=model_wd)
    
    initial_mass = calc_initial_mass(final_mass,e_final_mass,n_mc)
    
    ms_age = calc_ms_age(initial_mass,feh=feh,vvcrit=vvcrit)
    
    final_mass_median = np.ones(N)*np.nan
    final_mass_std = np.ones(N)*np.nan
    initial_mass_median = np.ones(N)*np.nan
    initial_mass_std = np.ones(N)*np.nan
    cooling_age_median = np.ones(N)*np.nan
    cooling_age_std = np.ones(N)*np.nan
    ms_age_median = np.ones(N)*np.nan
    ms_age_std = np.ones(N)*np.nan
    total_age_median = np.ones(N)*np.nan
    total_age_std = np.ones(N)*np.nan
    
    for i in range(N):
        final_mass_median[i] = np.nanmedian(final_mass2[i])
        final_mass_std[i] = np.nanstd(final_mass2[i])
        initial_mass_median[i] = np.nanmedian(initial_mass[i])
        initial_mass_std[i] = np.nanstd(initial_mass[i])
        cooling_age_median[i] = np.nanmedian(cooling_age[i])
        cooling_age_std[i] = np.nanstd(cooling_age[i])
        ms_age_median[i] = np.nanmedian(ms_age[i])
        ms_age_std[i] = np.nanstd(ms_age[i])
        total_age_median[i] = np.nanmedian(cooling_age[i] + ms_age[i])
        total_age_std[i] = np.nanstd(cooling_age[i] + ms_age[i])
    
    return total_age_median,total_age_std


'''
wd = Table.read('/Users/rociokiman/Documents/M-dwarfs-Age-Activity-Relation/Catalogs/WD_M_Pairs_new.csv')

logg = wd['loggH_wd']#np.array([7.966909,9.074503])
e_logg = wd['e_loggH_wd']#np.array([0.09473,0.62385])

teff = wd['TeffH_wd']#np.array([6022.601071,13065.57019])
e_teff = wd['e_TeffH_wd']#np.array([154.752034,4909.612193])

final_mass = wd['MassH_wd']#np.array([0.568221,1.228519])
e_final_mass = wd['e_MassH_wd']#np.array([0.252533,0.29186])



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