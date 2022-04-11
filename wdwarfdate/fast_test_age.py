import numpy as np
from scipy import interpolate
import inspect
import os
from astropy.table import Table
from .extra_func import calc_single_star_params
from .ifmr import calc_initial_mass
from .cooling_age import calc_cooling_age
from .ms_age import calc_ms_age


def estimate_parameters_fast_test(teff, e_teff, logg, e_logg, model_ifmr,
                                  model_wd, feh, vvcrit, n_mc, max_age,
                                  warning=1):

    # Set up the distribution of teff and logg
    log_cooling_age_dist = []
    final_mass_dist = []
    initial_mass_dist = []
    log_ms_age_dist = []
    log_total_age_dist = []


    for teff_i, e_teff_i, logg_i, e_logg_i,i in zip(teff, e_teff,
                                                    logg, e_logg,
                                                    range(len(teff))):
        if np.isnan(teff_i + e_teff_i + logg_i + e_logg_i):
            log_cooling_age_dist.append(np.array([np.nan]))
            final_mass_dist.append(np.array([np.nan]))
            initial_mass_dist.append(np.array([np.nan]))
            log_ms_age_dist.append(np.array([np.nan]))
            log_total_age_dist.append(np.array([np.nan]))
        else:
            approx = calc_single_star_params(teff_i, logg_i, model_wd,
                                             model_ifmr, feh, vvcrit)
            cool_age_i, final_mass_i, initial_mass_i, ms_age_i = approx

            if np.isnan(final_mass_i) or cool_age_i < np.log10(3e5):
                if warning == 1:
                    print("Warning: Effective temperature and/or surface " +
                          "gravity are outside of the allowed values of " +
                          f"the model. Teff = {np.round(teff_i, 2)} K, " +
                          f"logg = {np.round(logg_i, 2)}")
                log_cooling_age_dist.append(np.array([np.nan]))
                final_mass_dist.append(np.array([np.nan]))
                initial_mass_dist.append(np.array([np.nan]))
                log_ms_age_dist.append(np.array([np.nan]))
                log_total_age_dist.append(np.array([np.nan]))
            elif np.isnan(initial_mass_i):
                if warning == 1:
                    print("Warning: Final mass is outside the range allowed " +
                          "by the IFMR. Cannot estimate initial mass, main " +
                          "sequence age or total age. " +
                          f"Teff = {np.round(teff_i, 2)} K, " +
                          f"logg = {np.round(logg_i, 2)}, " +
                          f"Final mass ~ {np.round(final_mass_i, 2)} Msun ")
                teff_dist = np.random.normal(teff_i, e_teff_i, n_mc)
                logg_dist = np.random.normal(logg_i, e_logg_i, n_mc)
                res = calc_cooling_age(teff_dist, logg_dist, model=model_wd)
                log_cooling_age_dist_i, final_mass_dist_i = res

                min_cool_age = np.log10(0.3 * 1e6)
                mask_cool_age = log_cooling_age_dist_i > min_cool_age
                log_cooling_age_dist.append(log_cooling_age_dist_i[mask_cool_age])
                final_mass_dist.append(final_mass_dist_i[mask_cool_age])
                initial_mass_dist.append(np.array([np.nan]))
                log_ms_age_dist.append(np.array([np.nan]))
                log_total_age_dist.append(np.array([np.nan]))
            elif np.isnan(ms_age_i):
                if warning == 1:
                    print("Warning: Initial mass is outside of the range " +
                          "allowed by the MIST isochrones. Cannot estimate " +
                          "main sequence age or total age. " +
                          f"Teff = {np.round(teff_i, 2)} K, " +
                          f"logg = {np.round(logg_i, 2)}, " +
                          f"Initial mass ~ {np.round(initial_mass_i, 2)} Msun")
                teff_dist = np.random.normal(teff_i, e_teff_i, n_mc)
                logg_dist = np.random.normal(logg_i, e_logg_i, n_mc)
                res = calc_cooling_age(teff_dist, logg_dist, model=model_wd)
                log_cooling_age_dist_i, final_mass_dist_i = res
                ini_mass_dist_i = calc_initial_mass(model_ifmr,
                                                    final_mass_dist_i)
                ms_age_dist_i = calc_ms_age(ini_mass_dist_i, feh, vvcrit)
                log_tot_age_dist_i = np.log10(10**ms_age_dist_i
                                              + 10**log_cooling_age_dist_i)
                mask_age = (log_tot_age_dist_i < max_age)
                min_cool_age = np.log10(0.3*1e6)
                mask_cool_age = log_cooling_age_dist_i > min_cool_age
                mask_age = mask_age * mask_cool_age

                final_mass_dist.append(final_mass_dist_i[mask_age])
                log_cooling_age_dist.append(log_cooling_age_dist_i[mask_age])
                initial_mass_dist.append(ini_mass_dist_i[mask_age])
                log_ms_age_dist.append(np.array([np.nan]))
                log_total_age_dist.append(np.array([np.nan]))
            else:
                teff_dist = np.random.normal(teff_i, e_teff_i, n_mc)
                logg_dist = np.random.normal(logg_i, e_logg_i, n_mc)
                res = calc_cooling_age(teff_dist, logg_dist, model=model_wd)
                log_cooling_age_dist_i, final_mass_dist_i = res
                ini_mass_dist_i = calc_initial_mass(model_ifmr,
                                                    final_mass_dist_i)
                ms_age_dist_i = calc_ms_age(ini_mass_dist_i, feh, vvcrit)
                log_tot_age_dist_i = np.log10(10**ms_age_dist_i
                                              + 10**log_cooling_age_dist_i)
                mask_age = (log_tot_age_dist_i < max_age)
                min_cool_age = np.log10(0.3 * 1e6)
                mask_cool_age = log_cooling_age_dist_i > min_cool_age
                mask_age = mask_age * mask_cool_age

                log_cooling_age_dist.append(log_cooling_age_dist_i[mask_age])
                final_mass_dist.append(final_mass_dist_i[mask_age])
                log_ms_age_dist.append(ms_age_dist_i[mask_age])
                initial_mass_dist.append(ini_mass_dist_i[mask_age])
                log_total_age_dist.append(log_tot_age_dist_i[mask_age])
    if len(log_cooling_age_dist)>1:
        return [np.array(log_cooling_age_dist, dtype=object),
                np.array(final_mass_dist, dtype=object),
                np.array(initial_mass_dist, dtype=object),
                np.array(log_ms_age_dist, dtype=object),
                np.array(log_total_age_dist, dtype=object)]
    else:
        return [np.array(log_cooling_age_dist),
                np.array(final_mass_dist),
                np.array(initial_mass_dist),
                np.array(log_ms_age_dist),
                np.array(log_total_age_dist)]

def calc_cooling_age_fast_test(teff_dist, logg_dist, within_limits, model):
    # Load cooling track for the model selected.
    if model == 'DA':
        path = 'Models/cooling_models/Thick_seq_020_130.csv'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath, format='csv')
    elif model == 'DB':
        path = 'Models/cooling_models/Thin_seq_020_130.csv'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath, format='csv')
    else:
        print('Please choose DA or DB for model_wd. For now using DA.')
        path = 'Models/cooling_models/Thick_seq_020_130.csv'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath, format='csv')

    model_teff = table_model['Teff']
    model_logg = table_model['Log(g)']
    model_age = np.array(
        [np.log10(x) if x > 0 else np.nan for x in table_model['Age']])
    model_mass = np.array(
        [np.log10(x) if x > 0 else np.nan for x in table_model['M/Msun']])

    mass_array = np.array([x for x in set(model_mass)])
    mass_array = np.sort(mass_array)
    age_array = np.array([np.nanmax(model_age[model_mass == x]) for x in mass_array])

    f_age_mass = interpolate.CubicSpline(mass_array, age_array)

    model_age_modif = model_age / f_age_mass(model_mass)

    # Interpolate model for cooling age and final mass from the cooling tracks
    f_cooling_age = interpolate.LinearNDInterpolator((model_logg, model_teff),
                                                     model_age_modif,
                                                     fill_value=np.nan,
                                                     rescale=True)
    f_final_mass = interpolate.LinearNDInterpolator((model_logg, model_teff),
                                                    model_mass,
                                                    fill_value=np.nan,
                                                    rescale=True)

    cooling_age_dist, final_mass_dist = [], []
    for logg_dist_i, teff_dist_i,i in zip(logg_dist, teff_dist,
                                          range(len(teff_dist))):
        if within_limits[i] == 0:
            cooling_age_dist.append(np.array([np.nan]))
            final_mass_dist.append(np.array([np.nan]))
        else:
            # Calculate final mass from teff and logg for star i
            fm = f_final_mass(logg_dist_i, teff_dist_i)
            mass_dist_i = np.array(10**fm)
            # Calculate cooling age from teff and logg for star i
            c = f_cooling_age(logg_dist_i, teff_dist_i)
            cooling_age_dist_i = np.array(c*f_age_mass(fm))
            # Append results to final list
            cooling_age_dist.append(cooling_age_dist_i)
            final_mass_dist.append(mass_dist_i)

    return np.array(cooling_age_dist), np.array(final_mass_dist)


def calc_ms_age_fast_test(initial_mass_dist, within_limits, feh, vvcrit):
    """
    Calculates a main sequence age distribution of the white dwarf's progenitor
    for each white dwarf from the initial mass distribution using
    MIST isochrones. This function is used in the 'fast_test' method.

    Parameters
    ----------
    initial_mass_dist : list of arrays. List of initial masses distributions
                        for each white dwarf progenitor.
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00'
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'

    Returns
    -------
    ms_age_dist : list of arrays. List of main sequence age distributions
                       for each white dwarf progenitor.
    """

    # Load isochrone model
    file_path = 'Models/MIST/MIST_v1.2_feh_'
    path = file_path + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS_sum.csv'
    path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
    filepath = os.path.join(path1, path)

    table_model = Table.read(filepath)

    model_initial_mass = table_model['initial_mass']
    model_ms_age = np.log10(table_model['ms_age'])

    # Interpolate model using isochrone values
    f_ms_age = interpolate.interp1d(model_initial_mass,
                                    model_ms_age, kind='cubic')

    # Replace with nan all the values of the initial_mass not included in
    # the interpolation model
    initial_mass_dist_copy = np.copy(initial_mass_dist)
    mask_nan = np.isnan(initial_mass_dist_copy)
    initial_mass_dist_copy[mask_nan] = 2
    mask = np.logical_or(np.min(model_initial_mass) > initial_mass_dist_copy,
                         np.max(model_initial_mass) < initial_mass_dist_copy)
    initial_mass_dist_copy[mask] = np.nan
    initial_mass_dist_copy[mask_nan] = np.nan

    # Use the interpolated model to calculate main sequence age
    ms_age_dist = []
    for i,initial_mass_dist_i in enumerate(initial_mass_dist_copy):
        if within_limits[i] == 0:
            ms_age_dist.append(np.array([np.nan]))
        else:
            ms_age_dist_i = np.array([f_ms_age(x) for x in initial_mass_dist_i])
            ms_age_dist.append(ms_age_dist_i)

    ms_age_dist = np.array(ms_age_dist)

    return np.array(ms_age_dist)


def calc_initial_mass_fast_test(model_ifmr, final_mass_dist, within_limits):
    """
    Uses different initial-final mass relations to calculte progenitor's mass
    from the white dwarf mass (final mass). This function is used in the
    'fast_test' method.

    Parameters
    ----------
    model_ifmr : string. Initial to final mass relation model. Can be
                 'Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                 'Salaris_2009', 'Williams_2009'.
    final_mass_dist : list of arrays. List of final mass distributions
                      for each white dwarf.


    Returns
    -------
    initial_mass_dist : list of arrays. List of initial mass distributions
                        for each white dwarf progenitor.
    """
    initial_mass_dist = []
    n_mc = len(final_mass_dist[0])

    if model_ifmr == 'Cummings_2018_MIST':
        '''
        Uses initial-final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        to calculate progenitor's mass from the white dwarf mass.
        '''
        for i, final_mass_dist_i in enumerate(final_mass_dist):
            if within_limits[i] == 0:
                initial_mass_dist.append(np.array([np.nan]))
            else:
                initial_mass_dist_i = np.ones(n_mc) * np.nan
                for j in range(n_mc):
                    fm_dist_j = final_mass_dist_i[j]
                    if (0.525 < fm_dist_j) and (fm_dist_j <= 0.717):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.489) / 0.08
                    elif (0.71695 < fm_dist_j) and (fm_dist_j <= 0.8572):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.184) / 0.187
                    elif (0.8562 < fm_dist_j) and (fm_dist_j <= 1.327):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.471) / 0.107
                initial_mass_dist.append(initial_mass_dist_i)
    elif model_ifmr == 'Cummings_2018_PARSEC':
        '''
        Uses initial-final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        to calculate progenitor's mass from the white dwarf mass.
        '''
        for i,final_mass_dist_i in enumerate(final_mass_dist):
            if within_limits[i] == 0:
                initial_mass_dist.append(np.array([np.nan]))
            else:
                initial_mass_dist_i = np.ones(n_mc) * np.nan
                for j in range(n_mc):
                    fm_dist_j = final_mass_dist_i[j]
                    if (0.515 < fm_dist_j) and (fm_dist_j <= 0.72):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.476) / 0.0873
                    elif (0.72 < fm_dist_j) and (fm_dist_j <= 0.87):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.210) / 0.181
                    elif (0.87 < fm_dist_j) and (fm_dist_j <= 1.2497):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.565) / 0.0835
                initial_mass_dist.append(initial_mass_dist_i)
    elif model_ifmr == 'Salaris_2009':
        '''
        Uses initial-final mass relation from 
        Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009)
        to calculte progenitor's mass from the white dwarf mass.
        '''
        for i,final_mass_dist_i in enumerate(final_mass_dist):
            if within_limits[i] == 0:
                initial_mass_dist.append(np.array([np.nan]))
            else:
                initial_mass_dist_i = np.ones(n_mc) * np.nan
                for j in range(n_mc):
                    fm_dist_j = final_mass_dist_i[j]
                    if (0.5588 <= fm_dist_j) and (fm_dist_j <= 0.867):
                        initial_mass_dist_i[j] = (fm_dist_j - 0.331) / 0.134
                    elif 0.867 < fm_dist_j:
                        initial_mass_dist_i[j] = (fm_dist_j - 0.679) / 0.047
                initial_mass_dist.append(initial_mass_dist_i)
    elif model_ifmr == 'Williams_2009':
        '''
        Uses initial-final mass relation from 
        Williams, K. A., et al., Astrophys. J. 693, 355–369 (2009).
        to calculte progenitor's mass from the white dwarf mass.

        Mfinal = 0.339 ± 0.015 + (0.129 ± 0.004)Minit ;
        '''
        for i,final_mass_dist_i in enumerate(final_mass_dist):
            if within_limits[i] == 0:
                initial_mass_dist.append(np.array([np.nan]))
            else:
                initial_mass_dist_i = np.ones(n_mc) * np.nan
                for j in range(n_mc):
                    fm_dist_j = final_mass_dist_i[j]
                    initial_mass_dist_i[j] = (fm_dist_j - 0.339) / 0.129
                initial_mass_dist.append(initial_mass_dist_i)

    initial_mass_dist = np.array(initial_mass_dist)

    return initial_mass_dist


def check_teff_logg(teff, e_teff, logg, e_logg, model_wd, model_ifmr,
                    feh, vvcrit):

    for teff_i, e_teff_i, logg_i, e_logg_i in zip(teff, e_teff, logg, e_logg):
        approx = calc_single_star_params(teff_i, logg_i, model_wd, model_ifmr,
                                         feh, vvcrit)
        cool_age, final_mass, initial_mass, ms_age = approx

        if np.isnan(final_mass + cool_age) or cool_age < 5.477: #log10(0.3*1e6)
            print("Warning: Effective temperature and/or surface " +
                  "gravity are outside of the allowed values of " +
                  f"the model. Teff = {np.round(teff_i,2)} K, " +
                  f"logg = {np.round(logg_i,2)}")
        elif ~np.isnan(final_mass) and np.isnan(initial_mass):
            print("Warning: Final mass is outside the range allowed " +
                  "by the IFMR. Cannot estimate initial mass, main " +
                  "sequence age or total age. " +
                  f"Teff = {np.round(teff_i,2)} K, "+
                  f"logg = {np.round(logg_i,2)}, " +
                  f"Final mass ~ {np.round(final_mass,2)} Msun ")
        elif (~np.isnan(final_mass + cool_age + initial_mass)
              and np.isnan(ms_age)):
            print("Warning: Initial mass is outside of the range " +
                  "allowed by the MIST isochrones. Cannot estimate " +
                  "main sequence age or total age. " +
                  f"Teff = {np.round(teff_i,2)} K, " +
                  f"logg = {np.round(logg_i,2)}, " +
                  f"Initial mass ~ {np.round(initial_mass,2)} Msun ")


