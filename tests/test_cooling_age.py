#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.cooling_age import calc_cooling_age, get_cooling_model
import numpy as np


def test_calc_cooling_age():
    n = 100
    teff_d = np.array([np.random.normal(10000, 100, n)])
    logg_d = np.array([np.random.normal(8, 0.1, n)])
    cooling_age_d, final_mass_d = calc_cooling_age(teff_d, logg_d, 'DA')
    assert len(cooling_age_d[0]) == n
    assert len(final_mass_d[0]) == n

    cooling_age_d2, final_mass_d2 = calc_cooling_age(teff_d, logg_d, 'DB')
    assert len(cooling_age_d2[0]) == n
    assert len(final_mass_d2[0]) == n


def test_get_cooling_model():
    # Data from http://www.montrealwhitedwarfdatabase.org/evolution.html
    teff = 4670.609661
    logg = 7.794246000000001
    final_mass = 0.451
    cooling_age = np.log10(4.440 * 1e9)

    # Define model with the same data base
    f_teff, f_logg, model_age, model_mass = get_cooling_model('DA')

    assert (np.isclose(f_teff(final_mass, cooling_age), teff, atol=120))
    assert (np.isclose(f_logg(final_mass, cooling_age), logg, atol=0.05))