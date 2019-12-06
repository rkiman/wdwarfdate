import numpy as np
from coolingage import calc_cooling_age
from final2initial_mass import calc_initial_mass
from ms_age import calc_ms_age
import matplotlib.pyplot as plt

logg = np.array([7.966909,9.074503])
e_logg = np.array([0.09473,0.62385])

teff = np.array([6022.601071,13065.57019])
e_teff = np.array([154.752034,4909.612193])

final_mass = np.array([0.568221,1.228519])
e_final_mass = np.array([0.252533,0.29186])

n_mc = 1000

cooling_age,final_mass2 = calc_cooling_age(teff,e_teff,logg,e_logg,n_mc,
                                           model='DA')

initial_mass = calc_initial_mass(final_mass,e_final_mass,n_mc)

ms_age = calc_ms_age(initial_mass,feh='p0.00',vvcrit='0.4')

plt.hist(initial_mass[1])
plt.show()

 
plt.hist(np.log10(ms_age[0]))
plt.hist(np.log10(ms_age[0]+cooling_age[0]))
plt.show()

plt.hist(np.log10(ms_age[1]))
plt.hist(np.log10(ms_age[1]+cooling_age[1]))
plt.show()