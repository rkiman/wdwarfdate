import numpy as np
from scipy import interpolate

def calc_cooling_age(teff,e_teff,logg,e_logg,n_mc,model,plot_fit=False):
    
    if(not isinstance(teff,np.ndarray)):
        teff = np.array([teff])
        e_teff = np.array([e_teff])
        logg = np.array([logg])
        e_logg = np.array([e_logg])
        
    if(model == 'DA'):
        table_model = np.loadtxt('/Users/rociokiman/Documents/wdwarfdate/Models/cooling_models/Table_DA')
        
    if(model == 'DB'):
        table_model = np.loadtxt('/Users/rociokiman/Documents/wdwarfdate/Models/cooling_models/Table_DB')
    
    model_T = table_model[:,0]
    model_logg = table_model[:,1]
    model_age = table_model[:,40]
    model_mass = table_model[:,2]
    
    grid_model_T=model_T.reshape(5,51)
    grid_model_logg=model_logg.reshape(5,51)
    grid_model_age=model_age.reshape(5,51)
    grid_model_mass=model_mass.reshape(5,51)
    
    
    f_cooling_age=interpolate.RectBivariateSpline(grid_model_logg[:,0],grid_model_T[0],grid_model_age)
    f_final_mass=interpolate.RectBivariateSpline(grid_model_logg[:,0],grid_model_T[0],grid_model_mass)

    N = len(teff)
    cooling_age,final_mass = [],[]
    
    for i in range(N):
        
        teff_array = np.random.normal(teff[i],e_teff[i],n_mc)
        logg_array = np.random.normal(logg[i],e_logg[i],n_mc)
        
        age_dist = np.array([f_cooling_age(logg_array_i,teff_array_i)[0][0] for logg_array_i,teff_array_i in zip(logg_array,teff_array)])
        
        mass_dist = np.array([f_final_mass(logg_array_i,teff_array_i)[0][0] for logg_array_i,teff_array_i in zip(logg_array,teff_array)])
        
        cooling_age.append(age_dist)
        final_mass.append(mass_dist)
    
    return np.array(cooling_age),np.array(final_mass)

'''
logg = 9.20
e_logg = 0.07

teff = 42700
e_teff = 800
n_mc = 2000

cooling_age,final_mass = calc_cooling_age(teff,e_teff,logg,e_logg,n_mc,model='DA',plot_fit=False)
print(cooling_age/1e6)
'''