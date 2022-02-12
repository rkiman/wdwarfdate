import numpy as np


def ifmr_bayesian(initial_mass, ifmr_model, min_initial_mass_mist,
                  max_initial_mass_mist):
    """
    Calculates a final mass from an initial mass using the initial-final mass
    model chosen. This function is used in the 'bayesian' method.

    Parameters
    ----------
    initial_mass : float. Initial mass of the progenitor of the white dwarf.
    ifmr_model : string. Initial to final mass relation model. Can be
                 'Marigo_2020', 'Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                 'Salaris_2009' or 'Williams_2009'.
    min_initial_mass_mist : float. Minimum initial mass in the MIST isochrone.
    max_initial_mass_mist : float. Maximum initial mass in the MIST isochrone.

    Returns
    -------
    final_mass : float. Final mass of the white dwarf calculated using the
                 initial to final mass model chosen.
    """
    # Initialize variables
    if not isinstance(initial_mass, np.ndarray):
        initial_mass = np.asarray(initial_mass)
    final_mass = np.copy(initial_mass) * np.nan
    final_mass = np.asarray(final_mass)

    if ifmr_model == 'Cummings_2018_MIST':
        '''
        Initial-Final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on MIST isochrones
        '''
        # mask1 = ((max([min_initial_mass_mist, 0.83]) <= initial_mass)
        #          * (initial_mass < 2.85))
        # mask1 = ((max([min_initial_mass_mist, 0.45]) <= initial_mass)
        #         * (initial_mass < 2.85))
        mask0 = ((max([min_initial_mass_mist, 0.45]) <= initial_mass)
                 * (initial_mass < 0.83))
        mask1 = ((0.83 <= initial_mass)
                 * (initial_mass < 2.85))
        mask2 = (2.85 <= initial_mass) * (initial_mass < 3.60)
        mask3 = ((3.60 <= initial_mass)
                 * (initial_mass < (min([max_initial_mass_mist, 8.2]))))  # 7.20]))))
        final_mass[mask0] = 0.5554  # 0.83 * 0.08 + 0.489
        final_mass[mask1] = initial_mass[mask1] * 0.08 + 0.489
        final_mass[mask2] = initial_mass[mask2] * 0.187 + 0.184
        final_mass[mask3] = initial_mass[mask3] * 0.107 + 0.471

    elif ifmr_model == 'Cummings_2018_PARSEC':
        '''
        Initial-Final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on PARSEC isochrones
        '''
        mask0 = ((max([min_initial_mass_mist, 0.45]) <= initial_mass)
                 * (initial_mass < 0.87))
        mask1 = (0.87 <= initial_mass) * (initial_mass < 2.8)
        mask2 = (2.8 <= initial_mass) * (initial_mass < 3.65)
        mask3 = ((3.65 <= initial_mass)
                 * (initial_mass < (min([max_initial_mass_mist, 8.20]))))  # 7.20]))))
        final_mass[mask0] = 0.551951  # 0.87*0.0873+0.476
        final_mass[mask1] = initial_mass[mask1] * 0.0873 + 0.476
        final_mass[mask2] = initial_mass[mask2] * 0.181 + 0.210
        final_mass[mask3] = initial_mass[mask3] * 0.0835 + 0.565
    elif ifmr_model == 'Salaris_2009':
        '''
        Initial-Final mass relation from 
        Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009).
        '''
        mask1 = ((max([min_initial_mass_mist, 1.7]) <= initial_mass)
                 * (initial_mass < 4))
        mask2 = (4 <= initial_mass) * (initial_mass < max_initial_mass_mist)
        final_mass[mask1] = initial_mass[mask1] * 0.134 + 0.331
        final_mass[mask2] = initial_mass[mask2] * 0.047 + 0.679
    elif ifmr_model == 'Williams_2009':
        '''
        Uses initial-final mass relation from 
        Williams, K. A., et al., Astrophys. J. 693, 355–369 (2009).
        to calculte progenitor's mass from the white dwarf mass.
        
        Mfinal = 0.339 ± 0.015 + (0.129 ± 0.004)Minit ;
        '''
        final_mass = 0.339 + 0.129 * initial_mass

    elif ifmr_model == 'Marigo_2020':
        '''
        Initial-Final mass relation from 
        Maringo, P., et al., Nature Astronomy, 4, 1102-1110 (2020) 
        for initial masses < 3.65 and from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on MIST isochrones for masses > 3.65.
        '''

        mask0 = ((max([min_initial_mass_mist, 0.45]) <= initial_mass)
                 * (initial_mass < 0.83))
        mask1 = (0.83 <= initial_mass) * (initial_mass <= 1.51)
        mask2 = (1.51 < initial_mass) * (initial_mass <= 1.845)
        mask3 = (1.845 < initial_mass) * (initial_mass <= 2.21)
        mask4 = (2.21 < initial_mass) * (initial_mass <= 3.65)
        mask5 = ((3.65 < initial_mass)
                 * (initial_mass < (min([max_initial_mass_mist, 7.20]))))
        final_mass[mask0] = 0.53249
        final_mass[mask1] = initial_mass[mask1] * 0.103 + 0.447
        final_mass[mask2] = initial_mass[mask2] * 0.399 + 0.001
        final_mass[mask3] = initial_mass[mask3] * (-0.342) + 1.367
        final_mass[mask4] = initial_mass[mask4] * 0.181 + 0.210
        final_mass[mask5] = initial_mass[mask5] * 0.107 + 0.471
    return final_mass


def calc_initial_mass(ifmr_model, final_mass):
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

    if ifmr_model == 'Marigo_2020':
        ifmr_model = 'Cummings_2018_MIST'
        print('Using Cummings_2018_MIST, because Marigo 2020 cannot be'
              + ' used in this direction.')
    # Initialize variables
    final_mass = np.asarray(final_mass)
    initial_mass = np.copy(final_mass) * np.nan
    initial_mass = np.asarray(initial_mass)

    if ifmr_model == 'Cummings_2018_MIST':
        '''
        Initial-Final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on MIST isochrones
        '''
        # lower limit extended
        mask0 = (0.525 <= final_mass) * (final_mass < 0.717)
        mask1 = (0.71695 <= final_mass) * (final_mass < 0.8572)
        mask2 = (0.8562 <= final_mass) * (final_mass < 1.327)

        initial_mass[mask0] = (final_mass[mask0] - 0.489) / 0.08
        initial_mass[mask1] = (final_mass[mask1] - 0.184) / 0.187
        initial_mass[mask2] = (final_mass[mask2] - 0.471) / 0.107
    elif ifmr_model == 'Cummings_2018_PARSEC':
        '''
        Initial-Final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on PARSEC isochrones
        '''
        # lower limit extended
        mask1 = (0.515 <= final_mass) * (final_mass < 0.72)
        mask2 = (0.72 <= final_mass) * (final_mass < 0.87)
        mask3 = (0.87 <= final_mass) * (final_mass < 1.2497)

        initial_mass[mask1] = (final_mass[mask1] - 0.476) / 0.0873
        initial_mass[mask2] = (final_mass[mask2] - 0.210) / 0.181
        initial_mass[mask3] = (final_mass[mask3] - 0.565) / 0.0835
    elif ifmr_model == 'Salaris_2009':
        '''
        Initial-Final mass relation from 
        Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009).
        '''
        mask1 = (0.5588 <= final_mass) * (final_mass < 0.867)
        mask2 = (0.867 <= final_mass)

        initial_mass[mask1] = (final_mass[mask1] - 0.331) / 0.134
        initial_mass[mask2] = (final_mass[mask2] - 0.679) / 0.047
    elif ifmr_model == 'Williams_2009':
        '''
        Uses initial-final mass relation from 
        Williams, K. A., et al., Astrophys. J. 693, 355–369 (2009).
        to calculte progenitor's mass from the white dwarf mass.

        Mfinal = 0.339 ± 0.015 + (0.129 ± 0.004)Minit ;
        '''

        initial_mass = (final_mass - 0.339) / 0.129

    return initial_mass

