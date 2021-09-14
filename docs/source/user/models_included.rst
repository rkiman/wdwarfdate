.. _Models included:

Models included
===============

In order to estimate an age for a white dwarf a chain of models have to be used. Here are the models included in *wdwarfdate*.

+-----------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| wdwarfdate                                                                                                                                                                                                                                  |
+=================+==================+========================================================================================================================================================================================================+
| Models Included | Cooling Models   | Cooling tracks from the Montreal White Dwarf Group available `online <http://www.astro.umontreal.ca/~bergeron/CoolingModels/>`_ (These limits are approximated, see Figure below for the true limits): |
|                 |                  |  - Thick H layer: :math:`2,500 \lesssim T_{\rm eff} \lesssim 150,000` K, :math:`7.0 \lesssim \log g \lesssim 9.0`                                                                                      |
|                 |                  |  - Thin H layer: :math:`3,250 \lesssim T_{\rm eff} \lesssim 150,000` K, :math:`7.0 \lesssim \log g \lesssim 9.0`                                                                                       |
|                 +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                 | IFMR             | - `Marigo et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020NatAs...4.1102M/abstract>`_                                                                                                             |
|                 |                  | - MIST and PARSEC based relations from `Cummings et al. (2018) <https://iopscience.iop.org/article/10.3847/1538-4357/aadfd6>`_                                                                         |
|                 |                  | - `Salaris et al. (2009) <https://ui.adsabs.harvard.edu/abs/2009ApJ...692.1013S/abstract>`_                                                                                                            |
|                 |                  | - `Williams et al. (2009) <https://iopscience.iop.org/article/10.1088/0004-637X/693/1/355>`_                                                                                                           |
|                 +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                 | Isochrones       | MESA Isochrones `available online <http://waps.cfa.harvard.edu/MIST/>`_:                                                                                                                               |
|                 |                  |  - :math:`{\rm Fe/H} = -4.00, -1.00, 0.00, 0.50`                                                                                                                                                       |
|                 |                  |  - :math:`{\rm v/vcrit} = 0.0, 0.4`                                                                                                                                                                    |
|                 |                  |  - :math:`\alpha/{\rm Fe} = 0`                                                                                                                                                                         |
+-----------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Constrains      | Assumptions      | - Single star evolution                                                                                                                                                                                |
|                 |                  | - C/O core                                                                                                                                                                                             |
+-----------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



The Figure below shows the cooling tracks used by *wdwarfdate* which indicate the limits in effective temperature and surface gravity in more detail for the models of thich and thin H layers (Kiman et al. in prep).

.. image:: cooling_seq.png
   :width: 600

