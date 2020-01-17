import numpy as np
from astropy.table import Table
from scipy import interpolate
#from make_models_fit import fit_eep_model

def calc_ms_age(initial_mass,feh,vvcrit):
    ms_age = []
    
    file = '/Users/rociokiman/Documents/wdwarfdate/Models/MIST/MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_EEPS_sum.csv'
    table_model = Table.read(file)
    
    model_initial_mass = table_model['initial_mass']
    model_ms_age = table_model['ms_age']
    
    f_ms_age = interpolate.interp1d(model_initial_mass, model_ms_age)
    for initial_mass_i in initial_mass:
        ms_age_i = np.array([f_ms_age(x) for x in initial_mass_i])
        ms_age.append(ms_age_i)
    return np.array(ms_age)
