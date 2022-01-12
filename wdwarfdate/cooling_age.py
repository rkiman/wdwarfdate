import numpy as np
from scipy import interpolate
import inspect
import os
from astropy.table import Table


def get_cooling_model(model_wd):
    """
    Interpolates a function which calculates effective temperature and
    surface gravity from final mass
    and cooling age. To interpolate uses cooling tracks
    from Bedard et al. (2020)
    available online http://www.astro.umontreal.ca/∼bergeron/CoolingModels/.

    Parameters
    ----------
    model_wd : string. Spectral type of the white dwarf 'DA' or 'DB'.

    Returns
    -------
    f_teff : interpolated function. Calculates effective temperature from
             a final mass and a cooling age.
    f_logg : interpolated function. Calculates surface gravity from
             a final mass and a cooling age.
    model_age : array. List of cooling ages used to create the f_teff and
                f_logg models.
    model_mass : array. List of final masses used to create the f_teff and
                f_logg models.
    """

    # Load cooling tracks depending on the model of white dwarf chosen
    if model_wd == 'DA':
        path = 'Models/cooling_models/Thick_seq_020_130.csv'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath, format='csv')
    elif model_wd == 'DB':
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
        [np.log10(x) if x > 0 else -1 for x in table_model['Age']])
    model_mass = table_model['M/Msun']

    # Interpolate a model to calculate teff and logg from cooling age and
    # final mass
    f_teff = interpolate.LinearNDInterpolator((model_mass, model_age),
                                              model_teff,
                                              fill_value=np.nan)
    f_logg = interpolate.LinearNDInterpolator((model_mass, model_age),
                                              model_logg, fill_value=np.nan)

    return f_teff, f_logg, model_age, model_mass


def get_cooling_model_grid(model_wd):
    """
    Interpolates a function which calculates effective temperature and
    surface gravity from final mass
    and cooling age. To interpolate uses cooling tracks
    from Bedard et al. (2020)
    available online http://www.astro.umontreal.ca/∼bergeron/CoolingModels/.

    Parameters
    ----------
    model_wd : string. Spectral type of the white dwarf 'DA' or 'DB'.

    Returns
    -------
    f_teff : interpolated function. Calculates effective temperature from
             a final mass and a cooling age.
    f_logg : interpolated function. Calculates surface gravity from
             a final mass and a cooling age.
    model_age : array. List of cooling ages used to create the f_teff and
                f_logg models.
    model_mass : array. List of final masses used to create the f_teff and
                f_logg models.
    """

    # Load cooling tracks depending on the model of white dwarf chosen
    if model_wd == 'DA':
        path = 'Models/cooling_models/Thick_seq_020_130.csv'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath, format='csv')
    elif model_wd == 'DB':
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
    model_age = np.array([np.log10(x) if x > 0
                          else -1 for x in table_model['Age']])
    model_mass = np.array([np.log10(x) for x in table_model['M/Msun']])

    mass_array = np.array([x for x in set(model_mass)])
    mass_array = np.sort(mass_array)
    age_array = np.array([np.nanmax(model_age[model_mass == x]) for x in mass_array])

    f_age_mass = interpolate.CubicSpline(mass_array, age_array)

    model_age_modif = model_age / f_age_mass(model_mass)

    f_teff_prime = interpolate.LinearNDInterpolator((model_mass, model_age_modif),
                                                    model_teff,
                                                    fill_value=np.nan, rescale=True)

    f_logg_prime = interpolate.LinearNDInterpolator((model_mass, model_age_modif),
                                                    model_logg,
                                                    fill_value=np.nan, rescale=True)

    def f_teff(mass, age):
        age_modif = age / f_age_mass(mass)
        return f_teff_prime(mass, age_modif)

    def f_logg(mass, age):
        age_modif = age / f_age_mass(mass)
        return f_logg_prime(mass, age_modif)

    return f_teff, f_logg


def calc_cooling_age(teff, logg, model):
    """
    Calculates cooling age and final mass of the white dwarf using cooling
    tracks from Bedard et al. (2020)
    available online http://www.astro.umontreal.ca/∼bergeron/CoolingModels/.

    Parameters
    ----------
    teff : number or arrays. List of effective temperature distributions
                for each white dwarf.
    logg : list of arrays. List of surface gravity distributions
                for each white dwarf.
    N : scalar, arraya. Total number of white dwarf.
    model : string. Spectral type of the white dwarf 'DA' or 'DB'.

    Returns
    -------
    cooling_age : list of arrays. List of cooling age distributions
                  for each white dwarf.
    final_mass : list of arrays. List of final mass distributions
                 for each white dwarf.
    """
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
    model_mass = table_model['M/Msun']

    # Interpolate model for cooling age and final mass from the cooling tracks
    f_cooling_age = interpolate.LinearNDInterpolator((model_logg, model_teff),
                                                     model_age,
                                                     fill_value=np.nan)
    f_final_mass = interpolate.LinearNDInterpolator((model_logg, model_teff),
                                                    model_mass,
                                                    fill_value=np.nan)
    # Use the interpolated model to calculate final mass and cooling age from
    # effective temperature and logg
    cooling_age = f_cooling_age(logg, teff)
    final_mass = f_final_mass(logg, teff)

    return cooling_age, final_mass
