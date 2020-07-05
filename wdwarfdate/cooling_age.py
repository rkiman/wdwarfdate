import numpy as np
from scipy import interpolate
import pkg_resources

def calc_cooling_age(teff_dist,logg_dist,n_mc,N,model,plot_fit=False):
        
    if(model == 'DA'):
        path = '../Models/cooling_models/Table_DA'
        filepath = pkg_resources.resource_filename(__name__, path)
        table_model = np.loadtxt(filepath)
        rows = 51
        
    if(model == 'DB'):
        path = '../Models/cooling_models/Table_DB'
        filepath = pkg_resources.resource_filename(__name__, path)
        table_model = np.loadtxt(filepath)
        rows = 62
    

    model_T = table_model[:,0]
    model_logg = table_model[:,1]
    model_age = table_model[:,40]
    model_mass = table_model[:,2]
    
    grid_model_T=model_T.reshape(5,rows)
    grid_model_logg=model_logg.reshape(5,rows)
    grid_model_age=model_age.reshape(5,rows)
    grid_model_mass=model_mass.reshape(5,rows)
    
    f_cooling_age=interpolate.RectBivariateSpline(grid_model_logg[:,0],grid_model_T[0],grid_model_age)
    f_final_mass=interpolate.RectBivariateSpline(grid_model_logg[:,0],grid_model_T[0],grid_model_mass)

    cooling_age_dist,final_mass_dist = [],[]
    
    for i in range(N):

        cooling_age_dist_i = np.array([f_cooling_age(logg_dist_j,teff_dist_j)[0][0] for logg_dist_j,teff_dist_j in zip(logg_dist[i],teff_dist[i])])
        
        mass_dist_i = np.array([f_final_mass(logg_dist_j,teff_dist_j)[0][0] for logg_dist_j,teff_dist_j in zip(logg_dist[i],teff_dist[i])])
        
        cooling_age_dist.append(cooling_age_dist_i)
        final_mass_dist.append(mass_dist_i)
            
    return np.array(cooling_age_dist),np.array(final_mass_dist)
