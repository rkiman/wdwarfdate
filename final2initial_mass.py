import numpy as np
#import matplotlib.pyplot as plt

def calc_initial_mass(final_mass,e_final_mass,n_mc):
    
    if(not isinstance(final_mass,np.ndarray)):
        final_mass = np.array([final_mass])
        e_final_mass = np.array([e_final_mass])
        
    initial_mass = []
    N = len(final_mass)

    for i in range(N):
        final_mass_array = np.random.normal(final_mass[i],e_final_mass[i],
                                            n_mc)      
        initial_mass_array = np.ones(n_mc)*np.nan
        for j in range(n_mc):
            if((0.5554 < final_mass_array[j]) and (final_mass_array[j] <= 0.717)):
                initial_mass_array[j] = (final_mass_array[j] - 0.489)/0.08
            elif((0.71695 < final_mass_array[j]) and (final_mass_array[j] <= 0.8572)):
                initial_mass_array[j] = (final_mass_array[j] - 0.184)/0.187
            elif((0.8562 < final_mass_array[j]) and (final_mass_array[j] <= 1.2414)):
                initial_mass_array[j] = (final_mass_array[j] - 0.471)/0.107
        
        initial_mass.append(initial_mass_array)

    return np.array(initial_mass)



#mass = calc_initial_mass(1.1,0.1,2000)
#plt.hist(mass[0])
#plt.show()