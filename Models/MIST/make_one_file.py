import numpy as np
from astropy.table import Table                                                               
import glob  

models = ['MIST_v1.2_feh_m4.00_afe_p0.0_vvcrit0.0_EEPS',
          'MIST_v1.2_feh_m4.00_afe_p0.0_vvcrit0.4_EEPS',
          'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS',
          'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS',
          'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_EEPS',
          'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_EEPS']

for model in models: 
    print(model)
    initial_mass = []
    ms_age = []                                                                      
    for file in list(glob.glob(model+'/*.txt')):
        table = np.loadtxt(file) 
        n = len(table[:,0])
        initial_mass.append(table[0,1])
        ms_age.append(table[n-1,0])
    summary = Table()
    summary['initial_mass'] = initial_mass
    summary['ms_age'] = ms_age
    summary.write(model+'_sum.csv')
    