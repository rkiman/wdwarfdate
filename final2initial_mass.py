import numpy as np
#import matplotlib.pyplot as plt

def calc_initial_mass(final_mass,n_mc):
        
    initial_mass = []

    for final_mass_i in final_mass:     
        initial_mass_array = np.ones(n_mc)*np.nan
        for j in range(n_mc):
            if((0.5554 < final_mass_i[j]) and (final_mass_i[j] <= 0.717)):
                initial_mass_array[j] = (final_mass_i[j] - 0.489)/0.08
            elif((0.71695 < final_mass_i[j]) and (final_mass_i[j] <= 0.8572)):
                initial_mass_array[j] = (final_mass_i[j] - 0.184)/0.187
            elif((0.8562 < final_mass_i[j]) and (final_mass_i[j] <= 1.2414)):
                initial_mass_array[j] = (final_mass_i[j] - 0.471)/0.107
        
        initial_mass.append(initial_mass_array)

    return np.array(initial_mass)



#mass = calc_initial_mass(1.1,0.1,2000)
#plt.hist(mass[0])
#plt.show()