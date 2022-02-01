import numpy as np
from astropy.table import Table
from scipy import interpolate
import os
import inspect


def get_isochrone_model(feh, vvcrit):
    """
    Interpolates MIST isochrones to get a function that gives initial mass
    as a function of main sequence age. This function is used in the 'bayesian'
    method.

    Parameters
    ----------
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00'
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'

    Returns
    -------
    f_initial_mass : interpolated function. Calculates initial mass from
                     a main sequence age.
    model_initial_mass : array. List of initial masses used to create the
                         f_initial_mass model.
    model_ms_age : array. List of main sequence ages used to create the
                   f_initial_mass model.
    """
    # Load isochrone
    file_path = 'Models/MIST/MIST_v1.2_feh_'
    path = file_path + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS_sum.csv'
    path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
    filepath = os.path.join(path1, path)

    table_model = Table.read(filepath)

    model_initial_mass = table_model['initial_mass']
    model_ms_age = np.log10(table_model['ms_age'])

    # Interpolate model from isochrone
    f_initial_mass = interpolate.interp1d(model_ms_age, model_initial_mass,
                                          fill_value=np.nan)

    return f_initial_mass, model_initial_mass, model_ms_age


def get_isochrone_model_grid(feh, vvcrit):
    """
    Interpolates MIST isochrones to get a function that gives initial mass
    as a function of main sequence age. This function is used in the 'bayesian'
    method.

    Parameters
    ----------
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00'
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'

    Returns
    -------
    f_initial_mass : interpolated function. Calculates initial mass from
                     a main sequence age.
    model_initial_mass : array. List of initial masses used to create the
                         f_initial_mass model.
    model_ms_age : array. List of main sequence ages used to create the
                   f_initial_mass model.
    """
    # Load isochrone
    file_path = 'Models/MIST/MIST_v1.2_feh_'
    path = file_path + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS_sum.csv'
    path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
    filepath = os.path.join(path1, path)

    table_model = Table.read(filepath)

    model_initial_mass = table_model['initial_mass']
    model_ms_age = np.log10(table_model['ms_age'])

    # Interpolate model from isochrone
    f_ms_age = interpolate.interp1d(model_initial_mass, model_ms_age,
                                    fill_value=np.nan)

    return f_ms_age, model_initial_mass, model_ms_age


def calc_ms_age(initial_mass, feh, vvcrit):
    """
    Calculates a main sequence age distribution of the white dwarf's progenitor
    for each white dwarf from the initial mass distribution using
    MIST isochrones. This function is used in the 'fast_test' method.

    Parameters
    ----------
    initial_mass : number or array. List of initial masses distributions
                        for each white dwarf progenitor.
    feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00'
          or 'p0.50'
    vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'

    Returns
    -------
    ms_age : number or array. List of main sequence age distributions
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

    initial_mass = np.asarray(initial_mass)
    ms_age = np.copy(initial_mass) * np.nan
    ms_age = np.asarray(ms_age)

    mask1 = np.min(model_initial_mass) > initial_mass
    mask2 = np.max(model_initial_mass) < initial_mass
    mask3 = np.logical_or(~mask1, ~mask2)

    ms_age[mask3] = f_ms_age(initial_mass[mask3])

    return ms_age

