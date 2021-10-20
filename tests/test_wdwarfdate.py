#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.wdwarfdate import WhiteDwarf
import numpy as np


def test_calc_wd_age():
    perc_unc = 1
    teff = 22607.142857142855
    teff_err = perc_unc * teff / 100
    logg = 8.971428571428572
    logg_err = perc_unc * logg / 100

    WD = WhiteDwarf(teff, teff_err, logg, logg_err,
                    model_wd='DA', feh='p0.00', vvcrit='0.0',
                    model_ifmr='Cummings_2018_MIST',
                    high_perc=84, low_perc=16, method='fast_test',
                    datatype='log', save_plots=False,
                    display_plots=False)

    WD.calc_wd_age()

    results = WD.results_fast_test
    fm_up = np.round(results['final_mass_median'][0]
                     + results['final_mass_err_high'][0], 1)
    fm_down = np.round(results['final_mass_median'][0]
                       - results['final_mass_err_low'][0], 1)
    cool_age_up = np.round(results['cooling_age_median'][0]
                           + results['cooling_age_err_high'][0], 1)
    cool_age_down = np.round(results['cooling_age_median'][0]
                             - results['cooling_age_err_low'][0], 1)

    # Check results agree with Montreal Group
    assert fm_up >= 1.2 >= fm_down
    assert cool_age_up >= 8.5 >= cool_age_down

    teff = np.array([22607.142857142855, 15571.42857142857])
    teff_err = perc_unc * teff / 100
    logg = np.array([8.971428571428572, 8.314285714285715])
    logg_err = perc_unc * logg / 100

    fm_mwdd = np.array([1.2, 0.8])
    cool_age_mwdd = np.array([8.5, 8.5])
    WD = WhiteDwarf(teff, teff_err, logg, logg_err,
                    model_wd='DA', feh='p0.00', vvcrit='0.0',
                    model_ifmr='Cummings_2018_MIST',
                    high_perc=84, low_perc=16, method='fast_test',
                    datatype='log', save_plots=False,
                    display_plots=False)

    WD.calc_wd_age()

    results = WD.results_fast_test
    for i, x, y in zip(range(2), fm_mwdd, cool_age_mwdd):
        fm_up = np.round(results['final_mass_median'][i]
                         + results['final_mass_err_high'][i], 1)
        fm_down = np.round(results['final_mass_median'][i]
                           - results['final_mass_err_low'][i], 1)
        cool_age_up = np.round(results['cooling_age_median'][i]
                               + results['cooling_age_err_high'][i], 1)
        cool_age_down = np.round(results['cooling_age_median'][i]
                                 - results['cooling_age_err_low'][i], 1)

        # Check results agree with Montreal Group
        assert fm_up >= x >= fm_down
        assert cool_age_up >= y >= cool_age_down
