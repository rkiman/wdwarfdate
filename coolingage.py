import numpy as np
from make_models_fit import fit_cooling_model

def calc_cooling_age(teff,e_teff,logg,e_logg,n_mc,model,plot_fit=False):
    
    if(not isinstance(teff,np.ndarray)):
        teff = np.array([teff])
        e_teff = np.array([e_teff])
        logg = np.array([logg])
        e_logg = np.array([e_logg])
        
    if(model == 'DA'):
        logg_model_unique,res_age_fit,res_mass_fit = fit_cooling_model(model='DA',plot_fit=plot_fit)
        
    if(model == 'DB'):
        logg_model_unique,res_age_fit,res_mass_fit = fit_cooling_model(model='DB',plot_fit=plot_fit)
    
    N = len(teff)
    cooling_age,final_mass = [],[]
    
    for i in range(N):
        
        teff_array = np.random.normal(teff[i],e_teff[i],n_mc)
        logg_array = np.random.normal(logg[i],e_logg[i],n_mc)
        
        mask_logg_unique = [np.argmin(abs(logg_model_unique-logg_j)) for logg_j in logg_array]
        
        res_age = res_age_fit[mask_logg_unique]
        res_mass = res_mass_fit[mask_logg_unique]
        
        age_dist = np.array([np.polyval(res_age_j,teff_array_j) 
                            for res_age_j,teff_array_j in zip(res_age,teff_array)])
        mass_dist = np.array([np.polyval(res_mass_j,teff_array_j) 
                             for res_mass_j,teff_array_j in zip(res_mass,teff_array)])
        
        cooling_age.append(age_dist)
        final_mass.append(mass_dist)
    
    return np.array(cooling_age),np.array(final_mass)
        
