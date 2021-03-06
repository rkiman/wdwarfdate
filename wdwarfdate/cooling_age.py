import numpy as np
from scipy import interpolate
import inspect
import os


def get_cooling_model(model_wd):
    """
    Interpolates a function which calculates effective temperature and
    surface gravity from final mass
    and cooling age. To interpolate uses cooling tracks
    from Bergeron et al. (1995)
    available online http://www.astro.umontreal.ca/∼bergeron/CoolingModels/.
    This function is used in the 'bayesian' method.

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
    if (model_wd == 'DA'):
        path = 'Models/cooling_models/Table_DA'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = np.loadtxt(filepath)

    if (model_wd == 'DB'):
        path = 'Models/cooling_models/Table_DB'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = np.loadtxt(filepath)

    model_teff = table_model[:, 0]
    model_logg = table_model[:, 1]
    model_age = np.array(
        [np.log10(x) if x > 0 else -1 for x in table_model[:, 43]])
    model_mass = table_model[:, 2]

    # Removing -inf from model because interpolation doesn't work with them
    # model_age[np.isinf(model_age)] = -1

    # Interpolate a model to calculate teff and logg from cooling age and
    # final mass
    f_teff = interpolate.LinearNDInterpolator((model_mass, model_age),
                                              model_teff,
                                              fill_value=np.nan)
    f_logg = interpolate.LinearNDInterpolator((model_mass, model_age),
                                              model_logg, fill_value=np.nan)

    return f_teff, f_logg, model_age, model_mass


def calc_cooling_age(teff_dist, logg_dist, N, model):
    """
    Calculates cooling age and final mass of the white dwarf using cooling
    tracks from from Bergeron et al. (1995)
    available online http://www.astro.umontreal.ca/∼bergeron/CoolingModels/.
    This function is used in the 'fast_test' method.

    Parameters
    ----------
    teff_dist : list of arrays. List of effective temperature distributions
                for each white dwarf.
    logg_dist : list of arrays. List of surface gravity distributions
                for each white dwarf.
    N : scalar, arraya. Total number of white dwarf.
    model : string. Spectral type of the white dwarf 'DA' or 'DB'.

    Returns
    -------
    cooling_age_dist : list of arrays. List of cooling age distributions
                       for each white dwarf.
    final_mass_dist : list of arrays. List of final mass distributions
                      for each white dwarf.
    """
    # Load cooling track for the model selected.
    if model == 'DA':
        path = 'Models/cooling_models/Table_DA'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = np.loadtxt(filepath)
        rows = 61

    if model == 'DB':
        path = 'Models/cooling_models/Table_DB'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = np.loadtxt(filepath)
        rows = 72

    model_teff = table_model[:, 0]
    model_logg = table_model[:, 1]
    model_age = table_model[:, 43]
    model_mass = table_model[:, 2]

    # Removing -inf from model because interpolation doesn't work with them
    model_age[np.isinf(model_age)] = -1

    grid_model_teff = model_teff.reshape(5, rows)
    grid_model_logg = model_logg.reshape(5, rows)
    grid_model_age = model_age.reshape(5, rows)
    grid_model_mass = model_mass.reshape(5, rows)

    # Interpolate model for cooling age and final mass from the cooling tracks
    f_cooling_age = interpolate.RectBivariateSpline(grid_model_logg[:, 0],
                                                    grid_model_teff[0],
                                                    grid_model_age)
    f_final_mass = interpolate.RectBivariateSpline(grid_model_logg[:, 0],
                                                   grid_model_teff[0],
                                                   grid_model_mass)

    # Use the interpolated model to calculate final mass and cooling age from
    # effective temperature and logg
    cooling_age_dist, final_mass_dist = [], []
    for i in range(N):
        c = [f_cooling_age(x, y)[0][0] for x, y in
             zip(logg_dist[i], teff_dist[i])]
        cooling_age_dist_i = np.array(c)
        fm = [f_final_mass(x, y)[0][0] for x, y in
              zip(logg_dist[i], teff_dist[i])]
        mass_dist_i = np.array(fm)

        cooling_age_dist.append(cooling_age_dist_i)
        final_mass_dist.append(mass_dist_i)

    return np.array(cooling_age_dist), np.array(final_mass_dist)
