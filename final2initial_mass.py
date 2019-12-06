import numpy as np

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
        mask1 = (0.56 < final_mass_array) * (final_mass_array <= 0.72)
        mask2 = (0.72 < final_mass_array) * (final_mass_array <= 0.86)
        mask3 = (0.86 < final_mass_array) * (final_mass_array <= 1.24)
        initial_mass_array[mask1] = (final_mass_array[mask1] - 0.489)/0.08
        initial_mass_array[mask2] = (final_mass_array[mask2] - 0.184)/0.187
        initial_mass_array[mask3] = (final_mass_array[mask3] - 0.471)/0.107
        
        initial_mass.append(initial_mass_array)

    return np.array(initial_mass)