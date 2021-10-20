#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
from astropy.table import Table
from scipy import interpolate


def calc_percentiles(ms_age, cooling_age, total_age, initial_mass,
                     final_mass, high_perc, low_perc):
    res_percentiles = []

    for dist in [ms_age, cooling_age, total_age, initial_mass, final_mass]:
        if all(np.isnan(dist)):
            dist_median = np.nan
            dist_err_low = np.nan
            dist_err_high = np.nan
        else:
            dist_median = np.nanpercentile(dist, 50)
            dist_err_low = dist_median - np.nanpercentile(dist, low_perc)
            dist_err_high = np.nanpercentile(dist, high_perc) - dist_median

        for x in [dist_median, dist_err_low, dist_err_high]:
            res_percentiles.append(x)

    return res_percentiles


def calc_dist_percentiles(dist, high_perc, low_perc):
    median = np.array([np.nanpercentile(x, 50) for x in dist])
    err_h = [np.nanpercentile(x, high_perc) for x in dist]
    err_l = [np.nanpercentile(x, low_perc) for x in dist]
    err_h, err_l = np.array(err_h), np.array(err_l)
    return median, err_h - median, median - err_l


def plot_distributions(ms_age, cooling_age, total_age,
                       initial_mass, final_mass, datatype,
                       results, display, save_plots, name='none'):
    title = r"${{{0:.2f}}}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12, 3))

    axs = [ax1, ax2, ax3, ax4, ax5]
    if datatype == 'Gyr':
        labels = ['MS Age (Gyr)', 'Cooling Age (Gyr)', 'Total Age (Gyr)',
                  r'Initial Mass (M$\odot$)', r'Final Mass (M$\odot$)']
    else:
        labels = [r'$\log _{10}($MS Age$/{\rm yr})$',
                  r'$\log _{10}($Cooling Age$/{\rm yr})$',
                  r'$\log _{10}($Total$/{\rm yr})$',
                  r'Initial Mass (M$\odot$)', r'Final Mass (M$\odot$)']

    distributions = [ms_age, cooling_age, total_age, initial_mass, final_mass]

    for ax, label, dist, i in zip(axs, labels, distributions,
                                  np.arange(0, 15, 3)):
        ax.hist(dist[~np.isnan(dist)], bins=20)
        ax.axvline(x=results[i], color='k')
        ax.axvline(x=results[i] - results[i + 1], color='k', linestyle='--')
        ax.axvline(x=results[i] + results[i + 2], color='k', linestyle='--')
        ax.set_xlabel(label)
        ax.yaxis.set_visible(False)
        if any(np.array([np.round(results[i], 2), np.round(results[i + 1], 2),
                         np.round(results[i + 2], 2)]) == 0):
            dec_num = 2
            while any(np.array([np.round(results[i], dec_num),
                                np.round(results[i + 1], dec_num),
                                np.round(results[i + 2], dec_num)]) == 0):
                dec_num += 1
            title2 = r"${{{0:." + str(dec_num) + "f}}}_{{-{1:." + str(dec_num) + "f}}}^{{+{2:." + str(dec_num) + "f}}}$"
            ax.set_title(title2.format(np.round(results[i], dec_num),
                                       np.round(results[i + 1], dec_num),
                                       np.round(results[i + 2], dec_num)))
        else:
            ax.set_title(title.format(np.round(results[i], 2),
                                      np.round(results[i + 1], 2),
                                      np.round(results[i + 2], 2)))

    plt.tight_layout()
    if name == 'none':
        print('Please provide a name for the file, but for now we will save ' +
              'it as none')
    if save_plots:
        plt.savefig(name + '_distributions.png', dpi=300)
    if display:
        plt.show()
    plt.close(f)


def check_ranges(teff, logg, model):
    # Load cooling track for the model selected.
    if model == 'DA':
        path = 'Models/cooling_models/Thick_seq_020_130.fits'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath)

    if model == 'DB':
        path = 'Models/cooling_models/Thin_seq_020_130.fits'
        path1 = os.path.dirname(inspect.getfile(inspect.currentframe()))
        filepath = os.path.join(path1, path)
        table_model = Table.read(filepath)

    model_teff = table_model['Teff']
    model_logg = table_model['Log(g)']
    model_age = table_model['Age']
    model_mass = table_model['M/Msun']

    # Interpolate model for cooling age and final mass from the cooling tracks
    f_cooling_age = interpolate.LinearNDInterpolator((model_logg, model_teff),
                                                     model_age,
                                                     fill_value=np.nan)
    if ~(f_cooling_age(logg, teff) > 3e5):
        print(
            f"Warning: Effective temperature and surface temperature ({teff} and {logg}) "
            "are outside the range covered "
            "by the cooling tracks from "
            "http://www.astro.umontreal.ca/~bergeron/CoolingModels/")
