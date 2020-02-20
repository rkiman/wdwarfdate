import numpy as np
from astropy.table import Table
from scipy import interpolate
import matplotlib.pyplot as plt
#from make_models_fit import fit_eep_model

def calc_ms_age(initial_mass_dist,feh,vvcrit):
    ms_age_dist = []
    
    file = '/Users/rociokiman/Documents/wdwarfdate/Models/MIST/MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_EEPS_sum.csv'
    table_model = Table.read(file)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = table_model['ms_age']

    f_ms_age = interpolate.interp1d(model_initial_mass, model_ms_age, kind='cubic')
    '''
    x=np.linspace(0.1,1,100)
    plt.loglog(model_initial_mass,model_ms_age,'.')
    plt.loglog(x,f_ms_age(x),'-r')
    plt.xlim(0.01,100)
    plt.show()
    '''
    for initial_mass_dist_i in initial_mass_dist:
        ms_age_dist_i = np.array([f_ms_age(x) for x in initial_mass_dist_i])
        ms_age_dist.append(ms_age_dist_i)
    return np.array(ms_age_dist)
