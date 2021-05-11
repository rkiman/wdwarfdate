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
                 'Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                 'Salaris_2009' or 'Williams_2009'.
    min_initial_mass_mist : float. Minimum initial mass in the MIST isochrone.
    max_initial_mass_mist : float. Maximum initial mass in the MIST isochrone.

    Returns
    -------
    final_mass : float. Final mass of the white dwarf calculated using the
                 initial to final mass model chosen.
    """
    # Initialize variables
    initial_mass = np.asarray(initial_mass)
    final_mass = np.copy(initial_mass) * np.nan
    final_mass = np.asarray(final_mass)

    if ifmr_model == 'Cummings_2018_MIST':
        '''
        Initial-Final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on MIST isochrones
        '''
        mask1 = (min_initial_mass_mist <= initial_mass) * (initial_mass < 2.85)
        mask2 = (2.85 <= initial_mass) * (initial_mass < 3.60)
        mask3 = (3.60 <= initial_mass) * (initial_mass < max_initial_mass_mist)
        final_mass[mask1] = initial_mass[mask1] * 0.08 + 0.489
        final_mass[mask2] = initial_mass[mask2] * 0.187 + 0.184
        final_mass[mask3] = initial_mass[mask3] * 0.107 + 0.471

    elif ifmr_model == 'Cummings_2018_PARSEC':
        '''
        Initial-Final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        based on PARSEC isochrones
        '''
        mask1 = (0.87 <= initial_mass) * (initial_mass < 2.8)
        mask2 = (2.8 <= initial_mass) * (initial_mass < 3.65)
        mask3 = (3.65 <= initial_mass) * (initial_mass < 8.20)
        final_mass[mask1] = initial_mass[mask1] * 0.0873 + 0.476
        final_mass[mask2] = initial_mass[mask2] * 0.181 + 0.210
        final_mass[mask3] = initial_mass[mask3] * 0.0835 + 0.565
    elif ifmr_model == 'Salaris_2009':
        '''
        Initial-Final mass relation from 
        Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009).
        '''
        mask1 = (1.7 <= initial_mass) * (initial_mass < 4)
        mask2 = (4 <= initial_mass)
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
        mask1 = (min_initial_mass_mist <= initial_mass) * (initial_mass <= 1.51)
        mask2 = (1.51 < initial_mass) * (initial_mass <= 1.845)
        mask3 = (1.845 < initial_mass) * (initial_mass <= 2.21)
        mask4 = (2.21 < initial_mass) * (initial_mass <= 3.65)
        mask5 = (3.65 < initial_mass) * (initial_mass < max_initial_mass_mist)
        final_mass[mask1] = initial_mass[mask1] * 0.103 + 0.447
        final_mass[mask2] = initial_mass[mask2] * 0.399 + 0.001
        final_mass[mask3] = initial_mass[mask3] * (-0.342) + 1.367
        final_mass[mask4] = initial_mass[mask4] * 0.181 + 0.210
        final_mass[mask5] = initial_mass[mask5] * 0.107 + 0.471
    return final_mass


def calc_initial_mass(model_ifmr, final_mass_dist):
    """
    Uses different initial-final mass relations to calculte progenitor's mass
    from the white dwarf mass (final mass). This function is used in the
    'fast_test' method.

    Parameters
    ----------
    model_ifmr : string. Initial to final mass relation model. Can be
                 'Cummings_2018_MIST', 'Cummings_2018_PARSEC'
                 or 'Salaris_2009'.
    final_mass_dist : list of arrays. List of final mass distributions
                      for each white dwarf.


    Returns
    -------
    initial_mass_dist : list of arrays. List of initial mass distributions
                        for each white dwarf progenitor.
    """
    initial_mass_dist = []
    n_mc = len(final_mass_dist[0])

    if model_ifmr == 'Cummings_2018_MIST':
        '''
        Uses initial-final mass relation from 
        Cummings, J. D., et al., Astrophys. J. 866, 21 (2018)
        to calculte progenitor's mass from the white dwarf mass.
        '''
        for final_mass_dist_i in final_mass_dist:
            initial_mass_dist_i = np.ones(n_mc) * np.nan
            for j in range(n_mc):
                fm_dist_j = final_mass_dist_i[j]
                if ((0.5554 < fm_dist_j) and (fm_dist_j <= 0.717)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.489) / 0.08
                elif ((0.71695 < fm_dist_j) and (fm_dist_j <= 0.8572)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.184) / 0.187
                elif ((0.8562 < fm_dist_j) and (fm_dist_j <= 1.2414)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.471) / 0.107
                else:
                    0

            initial_mass_dist.append(initial_mass_dist_i)
    elif model_ifmr == 'Salaris_2009':
        '''
        Uses initial-final mass relation from 
        Salaris, M., et al., Astrophys. J. 692, 1013–1032 (2009)
        to calculte progenitor's mass from the white dwarf mass.
        '''
        print('Using Salaris 2009 IFMR')
        for final_mass_dist_i in final_mass_dist:
            initial_mass_dist_i = np.ones(n_mc) * np.nan
            for j in range(n_mc):
                fm_dist_j = final_mass_dist_i[j]
                if ((0.5588 <= fm_dist_j) and (fm_dist_j <= 0.867)):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.331) / 0.134
                elif (0.867 < fm_dist_j):
                    initial_mass_dist_i[j] = (fm_dist_j - 0.679) / 0.047
                else:
                    0

            initial_mass_dist.append(initial_mass_dist_i)
    elif model_ifmr == 'Williams_2009':
        '''
        Uses initial-final mass relation from 
        Williams, K. A., et al., Astrophys. J. 693, 355–369 (2009).
        to calculte progenitor's mass from the white dwarf mass.
        
        Mfinal = 0.339 ± 0.015 + (0.129 ± 0.004)Minit ;
        '''
        print('Using Williams 2009 IFMR')

        initial_mass_dist = (final_mass_dist - 0.339) / 0.129

    initial_mass_dist = np.array(initial_mass_dist)

    # Remove all the values that are lower than the limit
    # of initial mass in isochrones
    mask_nan = np.isnan(initial_mass_dist)
    initial_mass_dist[mask_nan] = 5
    mask_neg = initial_mass_dist < 0.1
    initial_mass_dist[mask_neg] = np.nan
    initial_mass_dist[mask_nan] = np.nan

    return initial_mass_dist
