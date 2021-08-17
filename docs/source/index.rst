
wdwarfdate
==========

*wdwarfdate* is an open source code which estimates ages of white dwarfs in a bayesian framework. *wdwarfdate* runs a chain of models assuming single star evolution, to estimate ages of white dwarfs and their uncertainties from an effective temperature and a surface gravity. 

.. Contents:

Basic usage
-----------

Let's set up two white dwarfs with their effective temperature and surface gravity. These two white dwarfs parameters come from Cummings,J.D. et al. (2018).

.. code-block:: python

    import wdwarfdate
    import numpy as np

    #Define data for the white dwarfs
    teffs = np.array([19250,20250])
    teffs_err = np.array([500,850])
    loggs = np.array([8.16,8.526])
    loggs_err = np.array([0.084,0.126])

To estimate the ages of these two white dwarfs we need to use :func:`wdwarfdate.calc_wd_age`. This function will take the effective temperature (*teff*) and surface gravity (*logg*) and use one of two methods to estimate the ages: *bayesian* or *freq*. The *bayesian* method will run a Markov Chain Monte Carlo using `emcee <https://emcee.readthedocs.io/en/latest/>`_, until convergence to maximize a likelohood. The *freq* method will generate a gaussian distribution for *teff* and *logg* using the uncertainties as starndard deviation, and pass the full distribution through a chain of models to calculate the total age of the white dwarfs and other parameters described below. For more information about the two methods see Kiman et al. in prep. 

For quick estimation of a white dwarf age, we recommend using the *freq* method, but this method is likely to underestimate the uncertanties. For a better estimation of the age, use the *bayesian* method. To calculate the ages of the two white dwarfs we initiated above, for the *bayesian* method we can do something like

.. code-block:: python

    results_bayes = wdwarfdate.calc_wd_age(teffs,teffs_err,loggs,loggs_err,
                                           method='bayesian', datatype='Gyr',
                                           plot=True)

To calculate the ages of white dwarfs in the *freq* mode we can do

.. code-block:: python

    results_freq = wdwarfdate.calc_wd_age(teffs,teffs_err,loggs,loggs_err,
                                          method='freq',datatype='Gyr',
                                          return_distributions=True)


*wdwarfdate* allows us to select which models are going to go into the chain of models for the age estimation:
1) Cooling tracks for a DA or DB white dwarf
2) The initial-to-final mass relation
3) The parameter for the isochrone for the progenitor star 
As we did not specified the models in this example, the outputs are going to be estimated assuming these are DA white dwarfs, using the Cummings et al. 2018 MIST initial-to-final mass relation and assuming solar metallicity for the progenitor star. Also with the *datatype* option we can select the units of the resulting ages. 

The output :data:`results` of :func:`wdwarfdate.calc_wd_age` will be an astropy Table with one row for each teff and logg given and the following columns:

- :data:`results['ms_age_median']` Median values of main sequence age distribution of the progenitor of the white dwarf
- :data:`results['ms_age_err_low']` The difference between the median and the 16th percentile of the main sequence age distribution
- :data:`results['ms_age_err_high']` The difference between the 84th percentile and the median of the main sequence age distribution
- :data:`results['cooling_age_median']` Median values of cooling age distribution of the white dwarf
- :data:`results['cooling_age_err_low']` The difference between the median and the 16th percentile of the cooling age distribution
- :data:`results['cooling_age_err_high']` The difference between the 84th percentile and the median of the cooling age distribution
- :data:`results['total_age_median']` Median values of total age distribution of the white dwarf
- :data:`results['total_age_err_low']` The difference between the median and the 16th percentile of the total age distribution
- :data:`results['total_age_err_high']` The difference between the 84th percentile and the median of the total age distribution
- :data:`results['initial_mass_median']` Median values of initial mass distribution, meaning the mass of the progenitor
- :data:`results['initial_mass_err_low']` The difference between the median and the 16th percentile of the initial mass distribution
- :data:`results['initial_mass_err_high']` The difference between the 84th percentile and the median of the initial mass distribution
- :data:`results['final_mass_median']` Median values of final mass distribution, meaning the mass of the white dwarf 
- :data:`results['final_mass_err_low']` The difference between the median and the 16th percentile of the final mass distribution
- :data:`results['final_mass_err_high']`: The difference between the 84th percentile and the median of the final mass distribution

When we run the *bayesian* method, *wdwarfdate* will also save five files per star in a folder called results (which the code will create if it doesn't exist):

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corner_plot.png which contains the corner plot for the three variables the code samples: main sequence age, cooling age and delta m.

2. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_walkers.png which contains the traces for each walker to confirm convergence of the code.

3. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age) and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age).

4. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corr_time.png which contains the result of the autocorrelation time as explained in the `tutorial <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_ by Dan Foreman-Mackey.

5. teff_20250_logg_8.526_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST.txt: Contains the likelihood evaluations in each step of the MCMC for all the parameters we are interested in. We mentioned that the run of the MCMC is done in three stages: burn in, calculation of autocorrelation time and 1000 autocorrelation time steps. This file contains the likelihood evaluations of the last stage only. These are the columns: 'ln_ms_age', 'ln_cooling_age', 'ln_total_age, 'initial_mass' and 'final_mass'.

When we run the *freq* method, *wdwarfdate* will save one file:

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_freq_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age) and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age), but obtained with the *freq* method described above.

For more explanation on how to use *wdwarfdate* see the tutorials.

.. User Guide
.. ----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   user/models_included
   user/citation


.. Tutorials
.. ---------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/freq_method.ipynb
   tutorials/bayesian_method.ipynb



License
-------

MIT License

Copyright (c) 2020 Rocio Kiman

More information `here <https://github.com/rkiman/wdwarfdate/blob/master/LICENSE>`_
