import numpy as np
#import matplotlib.pyplot as plt

def calc_initial_mass(final_mass_dist,n_mc):
    '''
    Uses initial-final mass relation from Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
    to calculte progenitor's mass from the white dwarf mass.
    '''        
    initial_mass_dist = []

    for final_mass_dist_i in final_mass_dist:     
        initial_mass_dist_i = np.ones(n_mc)*np.nan
        for j in range(n_mc):
            if((0.5554 < final_mass_dist_i[j]) and (final_mass_dist_i[j] <= 0.717)):
                initial_mass_dist_i[j] = (final_mass_dist_i[j] - 0.489)/0.08
            elif((0.71695 < final_mass_dist_i[j]) and (final_mass_dist_i[j] <= 0.8572)):
                initial_mass_dist_i[j] = (final_mass_dist_i[j] - 0.184)/0.187
            elif((0.8562 < final_mass_dist_i[j]) and (final_mass_dist_i[j] <= 1.2414)):
                initial_mass_dist_i[j] = (final_mass_dist_i[j] - 0.471)/0.107
            else:
                0
        
        initial_mass_dist.append(initial_mass_dist_i)

    return np.array(initial_mass_dist)



#mass = calc_initial_mass(1.1,0.1,2000)
#plt.hist(mass[0])
#plt.show()