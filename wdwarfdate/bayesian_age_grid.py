import numpy as np
import random
from .ifmr import ifmr_bayesian


def get_idx_sample(posterior_grid):
    posterior_flat = posterior_grid.flatten()
    idx_list = np.arange(0, len(posterior_flat))
    mask_nan = ~np.isnan(posterior_flat)
    sample_idx = np.array(random.choices(idx_list[mask_nan],
                                         posterior_flat[mask_nan], k=262144))
    return sample_idx


def get_dist_parameters(params_grid, sample_idx):
    params_flat = params_grid.flatten()
    sample = params_flat[sample_idx]
    median = np.nanmedian(sample)
    err_high = np.nanpercentile(sample, 84) - median
    err_low = median - np.nanpercentile(sample, 16)
    return sample, median, err_high, err_low


def get_other_params(mi_sample, log10_tcool_sample, delta_m_sample, models):

    model_ifmr, isochrone_model, _= models
    f_ms_age, model_initial_mass, model_ms_age = isochrone_model
    min_initial_mass_mist = np.nanmin(model_initial_mass)
    max_initial_mass_mist = np.nanmax(model_initial_mass)

    mf_sample = ifmr_bayesian(mi_sample, model_ifmr, min_initial_mass_mist,
                              max_initial_mass_mist)
    mf_true_sample = mf_sample + delta_m_sample
    log10_tms_sample = f_ms_age(mi_sample)
    log10_ttot_sample = np.log10(10 ** log10_tcool_sample
                                 + 10 ** log10_tms_sample)
    return mf_true_sample, log10_tms_sample, log10_ttot_sample


def calc_posterior_grid(teff_i, e_teff_i, logg_i, e_logg_i, models, n_mi,
                        n_log10_tcool, n_delta, min_mi, max_mi,
                        min_log10_tcool, max_log10_tcool, max_log10_age):
    """
    Parameters
    ----------
    min_mi : scalar. Minimum initial mass in which to run the grid. The chosen value will be the maximum between
             min_mi and 0.1. Units of solar masses.
    max_mi : scalar. Maximum initial mass in which to run the grid. Units of solar masses.
    min_ttot : scalar. Minimum total age in which to run the grid. The chosen value will be the maximum between
               min_mi and 40 Myr. Units of log10(yr).
    max_ttot : scalar. Maximum total age in which to run the grid. The chosen value will be the minimum between
               max_ttot and 13.8 Gyr. Units of log10(yr).
    """
    model_ifmr, isochrone_model, cooling_models = models

    f_ms_age, model_initial_mass, model_ms_age = isochrone_model
    min_initial_mass_mist = np.nanmin(model_initial_mass)
    max_initial_mass_mist = np.nanmax(model_initial_mass)

    f_teff, f_logg = cooling_models

    # Defining grid axis
    mi = np.linspace(min_mi, max_mi, n_mi)
    log10_tcool = np.linspace(min_log10_tcool, max_log10_tcool,
                              n_log10_tcool)
    delta_m = np.linspace(-0.1, 0.1, n_delta)

    # Defining grid using axis
    grid = np.meshgrid(mi, log10_tcool, delta_m, indexing='ij')

    # Defining grid version of axis
    mi_grid = grid[0]
    log10_tcool_grid = grid[1]
    delta_m_grid = grid[2]

    # Setting log priors
    log_prior_mi = (-2.3) * np.log(mi_grid)  # IMF prior on mi
    log_prior_tcool = log10_tcool_grid * np.log(10)  # Constant SFH prior in ttot
    log_prior_delta_m = -0.5 * (delta_m_grid / 0.03) ** 2  # Prior Delta m

    # Calculate teff and logg using the grid parameters
    log10_tms_grid = f_ms_age(mi_grid)
    log10_ttot_grid = np.copy(log10_tcool_grid) * np.nan
    diff = np.log10(10 ** log10_tcool_grid + 10 ** log10_tms_grid)
    mask = diff < max_log10_age
    log10_ttot_grid[mask] = np.log10(diff[mask])
    mf_grid = ifmr_bayesian(mi_grid, model_ifmr,
                            min_initial_mass_mist, max_initial_mass_mist)
    log10_mf_true_grid = np.log10(mf_grid + delta_m_grid)
    mask_nan = np.isnan(log10_mf_true_grid + log10_ttot_grid)  # Make the interpolation of teff
    log10_mf_true_grid[mask_nan] = 1  # and logg faster by removing nans
    log10_tcool_grid[mask_nan] = 1

    teff = f_teff(log10_mf_true_grid, log10_tcool_grid)
    logg = f_logg(log10_mf_true_grid, log10_tcool_grid)

    # Calculate log likelihood
    log_likelihood = -0.5 * ((teff - teff_i) ** 2 / e_teff_i ** 2
                             + (logg - logg_i) ** 2 / (e_logg_i ** 2))
    log_likelihood[mask_nan] = -np.inf

    # Calculate log posterior
    log_post = log_likelihood + log_prior_tcool + log_prior_mi + log_prior_delta_m

    # Calculate and normalize posterior
    #practical_norm = np.nanmax(log_post)
    #log_post_practical_norm = log_post - practical_norm
    exp_posterior = np.exp(log_post)
    posterior = exp_posterior / np.nansum(exp_posterior)

    # Compile results
    params = [mi, log10_tcool, delta_m]
    params_grid = [mi_grid, log10_tcool_grid, delta_m_grid]

    return params, params_grid, posterior
