import numpy as np
#import matplotlib.pyplot as plt

def calc_initial_mass(model_ifmr,final_mass_dist,n_mc):
    '''
    Uses different initial-final mass relations
    to calculte progenitor's mass from the white dwarf mass.
    '''        
    initial_mass_dist = []
    
    if(model_ifmr == 'Cummings_2018_MIST'):
        '''
        Uses initial-final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        to calculte progenitor's mass from the white dwarf mass.
        '''  
        for final_mass_dist_i in final_mass_dist:     
            initial_mass_dist_i = np.ones(n_mc)*np.nan
            for j in range(n_mc):
                fm_dist_j = final_mass_dist_i[j]
                if((0.5554 < fm_dist_j) and (fm_dist_j <= 0.717)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.489)/0.08
                elif((0.71695 < fm_dist_j) and (fm_dist_j <= 0.8572)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.184)/0.187
                elif((0.8562 < fm_dist_j) and (fm_dist_j <= 1.2414)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.471)/0.107
                else:
                    0
            
            initial_mass_dist.append(initial_mass_dist_i)
    elif(model_ifmr == 'Salaris_2009'):
        '''
        Uses initial-final mass relation from 
        Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009)
        to calculte progenitor's mass from the white dwarf mass.
        ''' 
        print('Using Salaris 2009 IFMR')
        for final_mass_dist_i in final_mass_dist:     
            initial_mass_dist_i = np.ones(n_mc)*np.nan
            for j in range(n_mc):
                fm_dist_j = final_mass_dist_i[j]
                if((0.5588 <= fm_dist_j) and (fm_dist_j <= 0.867)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.331)/0.134
                elif(0.867 < fm_dist_j):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.679)/0.047
                else:
                    0
            
            initial_mass_dist.append(initial_mass_dist_i)
    elif(model_ifmr == 'Williams_2009'):
        '''
        Uses initial-final mass relation from 
        Williams, K. A., et al., Astrophys. J. 693, 355–369 (2009).
        to calculte progenitor's mass from the white dwarf mass.
        
        Mfinal = 0.339 ± 0.015 + (0.129 ± 0.004)Minit ;
        ''' 
        print('Using Williams 2009 IFMR')
        
        initial_mass_dist = (final_mass_dist - 0.339)/0.129

    
    initial_mass_dist = np.array(initial_mass_dist)
    
    #Remove all the values that are lower than the limit 
    #of initial mass in isochrones
    mask_neg = initial_mass_dist < 0.1 
    initial_mass_dist[mask_neg] = np.nan

    return initial_mass_dist
