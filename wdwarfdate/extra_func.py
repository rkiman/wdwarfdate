#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def calc_percentiles(ms_age, cooling_age, total_age, initial_mass,
                     final_mass, high_perc, low_perc):

    res_percentiles = []

    for distribution in [ms_age, cooling_age, total_age, initial_mass,
                         final_mass]:
        dist_median = np.nanpercentile(distribution, 50)
        dist_err_low = dist_median - np.nanpercentile(dist_median, low_perc)
        dist_err_high = np.nanpercentile(dist_median, high_perc) - dist_median

        for x in [dist_median, dist_err_low, dist_err_high]:
            res_percentiles.append(x)

    return res_percentiles


def plot_distributions(ms_age, cooling_age, total_age,
                       initial_mass, final_mass, datatype,
                       results, name='none'):

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

    for ax, label, dist, i in zip(axs, labels, distributions, range(5)):
        ax.hist(dist[~np.isnan(dist)], bins=20)
        ax.axvline(x=results[i], color='k')
        ax.axvline(x=results[i] - results[i+1], color='k', linestyle='--')
        ax.axvline(x=results[i] + results[i+2], color='k', linestyle='--')
        ax.set_xlabel(label)
        ax.set_title(title.format(results[i], results[i+1], results[i+2]))

    plt.tight_layout()
    if name == 'none':
        print('Please provide a name for the file, but for now we will save ' +
              'it as none')
    plt.savefig(name + '_distributions.png', dpi=300)
    plt.close(f)


def check_ranges(teff, logg, spt):
    if spt == 'DA':
        if np.logical_or(teff > 150000, teff < 2500):
            print(
                "Warning: Effective temperature is outside the range covered "
                "by the cooling models from "
                "http://www.astro.umontreal.ca/~bergeron/CoolingModels/ 2500 "
                "K < Teff < 150000 K")
        if np.logical_or(logg > 9, teff < 7):
            print(
                "Warning: Surface gravity is outside the range covered by the "
                "cooling models from "
                "http://www.astro.umontreal.ca/~bergeron/CoolingModels/ 7 < "
                "logg < 9")
    elif spt == 'DB':
        if np.logical_or(teff > 150000, teff < 3250):
            print(
                "Warning: Effective temperature is outside the range covered "
                "by the cooling models from "
                "http://www.astro.umontreal.ca/~bergeron/CoolingModels/ 3250 "
                "K < Teff < 150000 K")
        if np.logical_or(logg > 9, teff < 7):
            print(
                "Warning: Surface gravity is outside the range covered by the "
                "cooling models from "
                "http://www.astro.umontreal.ca/~bergeron/CoolingModels/ 7 < "
                "logg < 9")
