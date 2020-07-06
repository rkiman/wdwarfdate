
Documentation wdwarfdate
========================

To calculate white dwarf ages

.. Contents:

Basic usage
-----------

::
    import wdwarfdate

    #Define data for the white dwarf
    teff = np.array([19250,20250])
    teff_err = np.array([500,850])
    logg = np.array([8.16,8.526])
    logg_err = np.array([0.084,0.126])

    #If you have, define values to compare the results
    m_f = np.array([0.717,0.949])
    m_i = np.array([3.03,3.25])
    t_cool = np.array([8.060697840353612,8.298853076409706])
    t_ms = np.array([8.658011396657113,8.569373909615045])
    t_tot = np.array([8.755874855672491,8.755874855672491])

    data_comparison = [[t_ms[i],t_cool[i],t_tot[i],m_i[i],m_f[i]] for i in range(len(teff))]

    results = wdwarfdate.calc_wd_age(teff,teff_err,logg,logg_err,method='bayesian',
                                                        comparison = data_comparison)

User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install


Models Included
---------------

In order to estimate an age for a white dwarf a chain of models have to be used. Here are the models included in *wdwarfdate*.

.. toctree::
   :maxdepth: 2

   models/ifmr
   models/ms
   models/coolingage

Tutorials
---------

Some tutorials on how to use the code

License
-------

.. toctree::
   :maxdepth: 2
	
   lic/licensefile

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
