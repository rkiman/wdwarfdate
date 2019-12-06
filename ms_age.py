import numpy as np
from make_models_fit import fit_eep_model

def calc_ms_age(initial_mass,feh,vvcrit):
    ms_age = []
    for initial_mass_i in initial_mass:
        res_age = fit_eep_model(feh=feh,vvcrit=vvcrit)
        log_initial_mass = np.log10(initial_mass_i)
        log_ms_age = np.polyval(res_age,log_initial_mass)
        ms_age.append(np.exp(log_ms_age))
    return np.array(ms_age)
