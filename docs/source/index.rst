
wdwarfdate
==========

*wdwarfdate* is an open source Python code which estimates ages of white dwarfs in a Bayesian framework. *wdwarfdate* runs a chain of models assuming single star evolution, to estimate ages of white dwarfs and their uncertainties from an effective temperature and a surface gravity.

.. Contents:

Basic usage
-----------

Let's set up two white dwarfs with their effective temperature and surface gravity. These two white dwarfs parameters come from Cummings,J.D. et al. (2018). To run *wdwarfdate* we just need to initialize the object :func:`WhiteDwarf` with the parameters indicating the models we want to use. For details on these parameters go to the tutorials in this documentation.

.. code-block:: python

    import wdwarfdate
    import numpy as np

    #Define data for the white dwarfs
    teffs = np.array([19250,20250])
    teffs_err = np.array([500,850])
    loggs = np.array([8.16,8.526])
    loggs_err = np.array([0.084,0.126])

    WD = wdwarfdate.WhiteDwarf(teffs,teffs_err,loggs,loggs_err,
                               model_wd='DA',feh='p0.00',vvcrit='0.0',
                               model_ifmr = 'Cummings_2018_MIST',
                               high_perc = 84, low_perc = 16,
                               datatype='yr',
                               save_plots=False, display_plots=True)
    WD.calc_wd_age()

The function :func:`WD.calc_wd_age` will take the effective temperature (:math:`T_{\rm eff}`) and surface gravity (:math:`\log g`) given and run `emcee <https://emcee.readthedocs.io/en/latest/>`_ to sample the posterior given by the models, until convergence. To estimate the initial conditions, *wdwarfdate* runs the *fast-test* method, to obtain a quick answer for the parameter estimation. The *fast-test* method will generate a gaussian distribution for :math:`T_{\rm eff}` and :math:`\log g` using the uncertainties as standard deviation, and pass the full distribution through a chain of models to calculate the total age of the white dwarfs and other parameters described below. This method is available to run separately. For more information on how to run *wdwarfdate* see the tutorials and Kiman et al. in prep.

*wdwarfdate* allows us to select which models are going to go into the chain of models for the age estimation:
1. Cooling tracks for a DA or DB white dwarf.
2. The initial-to-final mass relation.
3. The parameter for the isochrone for the progenitor star .
For details on the models available see the page :ref:`Models included` in this documentation and Kiman et al. in prep.

The output is saved on the object, and can be accessed by doing :data:`WD.results`. It will be an `astropy Table <https://docs.astropy.org/en/stable/table/index.html>`_ with one row for each :math:`T_{\rm eff}` and :math:`\log g` given. The columns of the Table correspond to median, the difference between median and low percentile (16th unless indicated otherwise), and the difference between high percentile and median (84th unless indicated otherwise) for the following parameters:

- :data:`ms_age` Median values of main sequence age distribution of the progenitor of the white dwarf
- :data:`cooling_age` Median values of cooling age distribution of the white dwarf
- :data:`total_age` Median values of total age distribution of the white dwarf
- :data:`initial_mass` Median values of initial mass distribution, meaning the mass of the progenitor
- :data:`final_mass` Median values of final mass distribution, meaning the mass of the white dwarf

When estimate these parameters using *wdwarfdate*, it outputs several plot which are useful to visualize the results and to confirm the convergence of the algorithm. For a detailed description of the output plots, see the Tutorials in this documentation.


.. User Guide
.. ----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   user/models_included
   user/citation


.. _Tutorials:
.. Tutorials
.. ---------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/bayesian_method.ipynb
   tutorials/fast_test.ipynb



License
-------

MIT License

Copyright (c) 2020 Rocio Kiman

More information `here <https://github.com/rkiman/wdwarfdate/blob/master/LICENSE>`_
