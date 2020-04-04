#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.table import Table
from astropy.io import fits
import numpy as np
import wdwarfdate
import os

if not os.path.exists('results'):
    os.makedirs('results')
    
###---- Comparison to Gagné, J., et al., Astrophys. J. 861, L13 (2018)----###

if not os.path.exists('results/gagne'):
    os.makedirs('results/gagne')
    
data_gagne = [np.log10(44*1e6),np.log10((81)*1e6), np.log10(117*1e6), 7.8, 
              1.28]

r_gagne = wdwarfdate.calc_bayesian_wd_age(42700,800,9.20,0.07,n_mc=1000,
                                          model_wd='DA',feh='p0.00',
                                          vvcrit='0.0',
                                          model_ifmr = 'Cummings_2018_PARSEC',
                                          init_params = [], 
                                          comparison = data_gagne, n = 100, 
                                          high_perc = 84, low_perc = 16, 
                                          plot = True, save_dist = True,
                                          path = 'results/gagne/')

table_name = 'results/gagne_2018_results_bayesian.fits'

result_gagne_table = Table()
result_gagne_table['ms_age_median'] = np.array([r_gagne[0]])
result_gagne_table['ms_age_err_low'] = np.array([r_gagne[1]])
result_gagne_table['ms_age_err_high'] = np.array([r_gagne[2]])
result_gagne_table['cooling_age_median'] = np.array([r_gagne[3]])
result_gagne_table['cooling_age_err_low'] = np.array([r_gagne[4]])
result_gagne_table['cooling_age_err_high'] = np.array([r_gagne[5]])
result_gagne_table['total_age_median'] = np.array([r_gagne[6]])
result_gagne_table['total_age_err_low'] = np.array([r_gagne[7]])
result_gagne_table['total_age_err_high'] = np.array([r_gagne[8]])
result_gagne_table['initial_mass_median'] = np.array([r_gagne[9]])
result_gagne_table['initial_mass_err_low'] = np.array([r_gagne[10]])
result_gagne_table['initial_mass_err_high'] = np.array([r_gagne[11]])
result_gagne_table['final_mass_median'] = np.array([r_gagne[12]])
result_gagne_table['final_mass_err_low'] = np.array([r_gagne[13]])
result_gagne_table['final_mass_err_high'] = np.array([r_gagne[14]])

#Save results to plot in notebook
result_gagne_table.write(table_name, format='fits', overwrite = True)

###---- Comparison to Cummings, et al., Astrophys. J. 866, 21 (2018) ----###

if not os.path.exists('results/cummings'):
    os.makedirs('results/cummings')
    
cummings_2018 = Table.read('Catalogs/Cummings2018.csv',format='csv')

teff = cummings_2018['Teff']
teff_err = cummings_2018['Teff_err']
logg = cummings_2018['logg']
logg_err = cummings_2018['logg_err']
final_mass = cummings_2018['Mf']
final_mass_err = cummings_2018['Mf_err']
tcool = cummings_2018['Tcool']*1e6
tcool_err_high = cummings_2018['Tcool_err_high']*1e6
tcool_err_low = cummings_2018['Tcool_err_low']*1e6
initial_mass = cummings_2018['Mi_mist']
initial_mass_err_low = cummings_2018['Mi_mist_err_low']
initial_mass_err_high = cummings_2018['Mi_mist_err_high']
total_age = cummings_2018['Age_mist']*1e6
total_age_err = cummings_2018['Age_mist_err']*1e6

N = len(teff)
r_c_2018 = np.ones((N,15))*np.nan
model_ifmr = 'Cummings_2018_MIST'
for i in range(N):
    data_c_2018_i = [np.nan,np.log10(tcool[i]),np.log10(total_age[i]),
                     initial_mass[i],final_mass[i]]
    r_c_2018_i = wdwarfdate.calc_bayesian_wd_age(teff[i],teff_err[i], logg[i],
                                                 logg_err[i], n_mc=1000,
                                                 model_wd='DA', feh='p0.00',
                                                 vvcrit='0.0', 
                                                 model_ifmr = model_ifmr,
                                                 comparison = data_c_2018_i, 
                                                 n = 100, high_perc = 84, 
                                                 low_perc = 16, plot = True, 
                                                 save_dist = True, 
                                                 datatype = 'Gyr',
                                                 path = 'results/cummings/')
    r_c_2018[i,:] = r_c_2018_i

table_name = 'results/cummings_2018_results_bayesian.fits'

result_cummings_2018_table = Table()
result_cummings_2018_table['ms_age_median'] = r_c_2018[:,0]
result_cummings_2018_table['ms_age_err_low'] = r_c_2018[:,1]
result_cummings_2018_table['ms_age_err_high'] = r_c_2018[:,2]
result_cummings_2018_table['cooling_age_median'] = r_c_2018[:,3]
result_cummings_2018_table['cooling_age_err_low'] = r_c_2018[:,4]
result_cummings_2018_table['cooling_age_err_high'] = r_c_2018[:,5]
result_cummings_2018_table['total_age_median'] = r_c_2018[:,6]
result_cummings_2018_table['total_age_err_low'] = r_c_2018[:,7]
result_cummings_2018_table['total_age_err_high'] = r_c_2018[:,8]
result_cummings_2018_table['initial_mass_median'] = r_c_2018[:,9]
result_cummings_2018_table['initial_mass_err_low'] = r_c_2018[:,10]
result_cummings_2018_table['initial_mass_err_high'] = r_c_2018[:,11]
result_cummings_2018_table['final_mass_median'] = r_c_2018[:,12]
result_cummings_2018_table['final_mass_err_low'] = r_c_2018[:,13]
result_cummings_2018_table['final_mass_err_high'] = r_c_2018[:,14]

#Save results to plot in notebook
result_cummings_2018_table.write(table_name, format='fits', overwrite=True)

###---- Comparison to Garcés, et al., Astron. Astrophys. 531, 7 (2011) ----###

if not os.path.exists('results/garces'):
    os.makedirs('results/garces')
    
garces2011 = Table.read('Catalogs/Garces2011.csv')

mask_good = garces2011['Good'] == 1
garces2011 = garces2011[mask_good]

model_ifmr = 'Cummings_2018_MIST'
N = len(garces2011)

result_garces2011 = np.ones((N,15))*np.nan
for i in range(len(garces2011)):
    data_garces = [np.log10(garces2011['ms_age'][i]*1e9),
                   np.log10(garces2011['tcool'][i]*1e9),
                   np.log10(garces2011['Age'][i]*1e9),
                   garces2011['Minitial_mass'][i],garces2011['final_mass'][i]]
    r_g2011_i = wdwarfdate.calc_bayesian_wd_age(garces2011['Teff'][i],
                                                garces2011['e_Teff'][i],
                                                garces2011['logg'][i],
                                                garces2011['e_logg'][i],
                                                n_mc=1000,model_wd='DA',
                                                feh='p0.00', vvcrit='0.0',
                                                model_ifmr = model_ifmr,
                                                comparison = data_garces,
                                                n = 100, high_perc = 84, 
                                                low_perc = 16, plot = True,
                                                path='results/garces/')
    result_garces2011[i,:] = r_g2011_i
    
table_name = 'Garces2011_results_bayesian.fits'

result_garces2011_table = Table()
result_garces2011_table['ms_age_median'] = result_garces2011[:,0]
result_garces2011_table['ms_age_err_low'] = result_garces2011[:,1]
result_garces2011_table['ms_age_err_high'] = result_garces2011[:,2]
result_garces2011_table['cooling_age_median'] = result_garces2011[:,3]
result_garces2011_table['cooling_age_err_low'] = result_garces2011[:,4]
result_garces2011_table['cooling_age_err_high'] = result_garces2011[:,5]
result_garces2011_table['total_age_median'] = result_garces2011[:,6]
result_garces2011_table['total_age_err_low'] = result_garces2011[:,7]
result_garces2011_table['total_age_err_high'] = result_garces2011[:,8]
result_garces2011_table['initial_mass_median'] = result_garces2011[:,9]
result_garces2011_table['initial_mass_err_low'] = result_garces2011[:,10]
result_garces2011_table['initial_mass_err_high'] = result_garces2011[:,11]
result_garces2011_table['final_mass_median'] = result_garces2011[:,12]
result_garces2011_table['final_mass_err_low'] = result_garces2011[:,13]
result_garces2011_table['final_mass_err_high'] = result_garces2011[:,14]

#Save results to plot in notebook
result_garces2011_table.write(table_name, format = 'fits', overwrite = True)

###---- Comparison to BASE 9 code ----###

if not os.path.exists('results/base9'):
    os.makedirs('results/base9')
    
binaries = fits.open('Catalogs/VonHippel/mdwarf-wdwarf_binaries.fits')
check_age = Table.read('Catalogs/VonHippel/stats.summary.local',format='ascii')
log = Table.read('Catalogs/VonHippel/trans_log.csv')

gaia_id_binaries = binaries[1].data['Source_wd']
teff_binaries = binaries[1].data['TeffH_wd']
e_teff_binaries = binaries[1].data['e_TeffH_wd']
logg_binaries = binaries[1].data['loggH_wd']
e_logg_binaries = binaries[1].data['e_loggH_wd']
mass_binaries = binaries[1].data['MassH_wd']
e_mass_binaries = binaries[1].data['e_MassH_wd']
bp = binaries[1].data['phot_bp_mean_mag_wd']
rp = binaries[1].data['phot_rp_mean_mag_wd']
color_wd = bp - rp

gaia_id_log = np.array([x for x in log['Gaia_id']])
extra_id_log = np.array([x for x in log['extra_id']])

extra_id_check = np.array([float(x) for x in check_age['col1'][1:]])
total_age_check = np.array([float(x) for x in check_age['col6'][1:]])
e_total_age_check_lo = np.array([float(x) for x in check_age['col7'][1:]])
e_total_age_check_hi = np.array([float(x) for x in check_age['col8'][1:]])

cooling_age_check = np.array([float(x) for x in check_age['col36'][1:]])
e_cooling_age_check_lo = np.array([float(x) for x in check_age['col37'][1:]])
e_cooling_age_check_hi = np.array([float(x) for x in check_age['col38'][1:]])

ms_age_check = np.array([float(x) for x in check_age['col41'][1:]])
e_ms_age_check_lo = np.array([float(x) for x in check_age['col42'][1:]])
e_ms_age_check_hi = np.array([float(x) for x in check_age['col43'][1:]])

initial_mass_check = np.array([float(x) for x in check_age['col21'][1:]])
e_initial_mass_check_lo = np.array([float(x) for x in check_age['col22'][1:]])
e_initial_mass_check_hi = np.array([float(x) for x in check_age['col23'][1:]])

total_age = []
e_total_age_lo = []
e_total_age_hi = []
cooling_age = []
e_cooling_age_lo = []
e_cooling_age_hi = []
ms_age = []
e_ms_age_lo = []
e_ms_age_hi = []
initial_mass = []
e_initial_mass_lo = []
e_initial_mass_hi = []

for x in extra_id_log:
    mask = x == extra_id_check
    if(any(mask)):
        total_age.append(total_age_check[mask][0])
        e_total_age_lo.append(e_total_age_check_lo[mask][0])
        e_total_age_hi.append(e_total_age_check_hi[mask][0])
        cooling_age.append(cooling_age_check[mask][0])
        e_cooling_age_lo.append(e_cooling_age_check_lo[mask][0])
        e_cooling_age_hi.append(e_cooling_age_check_hi[mask][0])
        ms_age.append(ms_age_check[mask][0])
        e_ms_age_lo.append(e_ms_age_check_lo[mask][0])
        e_ms_age_hi.append(e_ms_age_check_hi[mask][0])
        initial_mass.append(initial_mass_check[mask][0])
        e_initial_mass_lo.append(e_initial_mass_check_lo[mask][0])
        e_initial_mass_hi.append(e_initial_mass_check_hi[mask][0])
    else:
        total_age.append(np.nan)
        e_total_age_lo.append(np.nan)
        e_total_age_hi.append(np.nan)
        cooling_age.append(np.nan)
        e_cooling_age_lo.append(np.nan)
        e_cooling_age_hi.append(np.nan)
        ms_age.append(np.nan)
        e_ms_age_lo.append(np.nan)
        e_ms_age_hi.append(np.nan)
        initial_mass.append(np.nan)
        e_initial_mass_lo.append(np.nan)
        e_initial_mass_hi.append(np.nan)

total_age = (10**np.array(total_age))/1e9
e_total_age_lo = total_age - (10**np.array(e_total_age_lo))/1e9
e_total_age_hi = (10**np.array(e_total_age_hi))/1e9 - total_age
cooling_age = (10**np.array(cooling_age))/1e9
e_cooling_age_lo = cooling_age - (10**np.array(e_cooling_age_lo))/1e9
e_cooling_age_hi = (10**np.array(e_cooling_age_hi))/1e9 - cooling_age
ms_age = (10**np.array(ms_age))/1e9
e_ms_age_lo = ms_age - (10**np.array(e_ms_age_lo))/1e9
e_ms_age_hi = (10**np.array(e_ms_age_hi))/1e9 - ms_age
initial_mass = np.array(initial_mass)
e_initial_mass_lo = initial_mass - np.array(e_initial_mass_lo)
e_initial_mass_hi = np.array(e_initial_mass_hi) - initial_mass

mask = ~np.isnan(teff_binaries+logg_binaries)

results_original_base9 = Table()

results_original_base9['teff'] = teff_binaries[mask]
results_original_base9['e_teff'] = e_teff_binaries[mask]
results_original_base9['logg'] = logg_binaries[mask]
results_original_base9['e_logg'] = e_logg_binaries[mask]
results_original_base9['total_age'] = total_age[mask]
results_original_base9['e_total_age_lo'] = e_total_age_lo[mask]
results_original_base9['e_total_age_hi'] = e_total_age_hi[mask]
results_original_base9['cooling_age'] = cooling_age[mask]
results_original_base9['e_cooling_age_lo'] = e_cooling_age_lo[mask]
results_original_base9['e_cooling_age_hi'] = e_cooling_age_hi[mask]
results_original_base9['ms_age'] = ms_age[mask]
results_original_base9['e_ms_age_lo'] = e_ms_age_lo[mask]
results_original_base9['e_ms_age_hi'] = e_ms_age_hi[mask]
results_original_base9['initial_mass'] = initial_mass[mask]
results_original_base9['e_initial_mass_lo'] = e_initial_mass_lo[mask]
results_original_base9['e_initial_mass_hi'] = e_initial_mass_hi[mask]

results_original_base9.write('Catalogs/VonHippel/results_compilation.fits',
                             format='fits', overwrite=True)

#Run wdwarfdate

teff_binaries = teff_binaries[mask]
e_teff_binaries = e_teff_binaries[mask]
logg_binaries = logg_binaries[mask]
e_logg_binaries = e_logg_binaries[mask]
ms_age = np.log10(ms_age[mask]*1e9)
cooling_age = np.log10(cooling_age[mask]*1e9)
total_age = np.log10(total_age[mask]*1e9)
initial_mass = initial_mass[mask]

N = len(teff_binaries)

result_base9 = np.ones((N,15))*np.nan
model_ifmr = 'Cummings_2018_MIST'
for i in range(N):
    data_base9 = [ms_age[i],cooling_age[i],total_age[i],initial_mass[i],np.nan]
    r_i = wdwarfdate.calc_bayesian_wd_age(teff_binaries[i],e_teff_binaries[i],
                                          logg_binaries[i],e_logg_binaries[i],
                                          n_mc=1000,model_wd='DA',feh='p0.00',
                                          vvcrit='0.0',model_ifmr = model_ifmr,
                                          comparison = data_base9, n = 100, 
                                          high_perc = 84, low_perc = 16, 
                                          plot = True, path = 'results/base9/')
    result_base9[i,:] = r_i
    
result_base9_table = Table()
result_base9_table['ms_age_median'] = result_base9[:,0]
result_base9_table['ms_age_err_low'] = result_base9[:,1]
result_base9_table['ms_age_err_high'] = result_base9[:,2]
result_base9_table['cooling_age_median'] = result_base9[:,3]
result_base9_table['cooling_age_err_low'] = result_base9[:,4]
result_base9_table['cooling_age_err_high'] = result_base9[:,5]
result_base9_table['total_age_median'] = result_base9[:,6]
result_base9_table['total_age_err_low'] = result_base9[:,7]
result_base9_table['total_age_err_high'] = result_base9[:,8]
result_base9_table['initial_mass_median'] = result_base9[:,9]
result_base9_table['initial_mass_err_low'] = result_base9[:,10]
result_base9_table['initial_mass_err_high'] = result_base9[:,11]
result_base9_table['final_mass_median'] = result_base9[:,12]
result_base9_table['final_mass_err_low'] = result_base9[:,13]
result_base9_table['final_mass_err_high'] = result_base9[:,14]

#Save results to plot in notebook
result_base9_table.write('results/base9_results_bayesian.fits', 
                         format ='fits', overwrite = True)
