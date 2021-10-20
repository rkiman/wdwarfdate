#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wdwarfdate.ifmr import ifmr_bayesian
from wdwarfdate.ifmr import calc_initial_mass
import numpy as np


def test_ifmr_bayesian_calc_initial_mass():
    for ifmr_model in ['Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                       'Salaris_2009', 'Williams_2009', 'Marigo_2020']:

        initial_mass = 3

        final_mass = ifmr_bayesian(initial_mass, ifmr_model, 0, 20)

        final_mass_dist = [np.ones(2) * final_mass, np.ones(2) * final_mass]
        if ifmr_model == 'Marigo_2020':
            ifmr_model_dummy = 'Cummings_2018_MIST'
        else:
            ifmr_model_dummy = ifmr_model
        initial_mass_2 = calc_initial_mass(ifmr_model_dummy, final_mass_dist)

        assert np.isclose(initial_mass, initial_mass_2[0][0], atol=1e-1)
