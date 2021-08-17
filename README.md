
wdwarfdate
==========

*wdwarfdate* is an open source code which estimates ages of white dwarf in a bayesian framework. *wdwarfdate* runs a chain of models assuming single star evolution, to estimate ages of white dwarfs and their uncertainties from an effective temperature and a surface gravity. Checkout the documentation for *wdwarfdate* [here](https://wdwarfdate.readthedocs.io/en/latest/) (under construction).


Installation
------------

To install *wdwarfdate* please do it from the source in GitHub. This can be done with the following lines:

```bash
    git clone https://github.com/rkiman/wdwarfdate.git
    cd wdwarfdate
    python setup.py install
```

**Dependences**

To run *wdwarfdate* the following packages are needed: [*NumPy*](https://numpy.org/), [*astropy*](https://www.astropy.org/), [*matplotlib*](https://matplotlib.org/), [*emcee*](https://emcee.readthedocs.io/en/latest/), [*corner*](https://corner.readthedocs.io/en/latest/), [*SciPy*](https://www.scipy.org/) and [*daft*](https://pypi.org/project/daft/). These can be installed with the following line:

```bash
    pip install numpy astropy matplotlib emcee corner scipy daft
```


Example usage
-------------

Getting right to it, let's make and example on how to estimate the ages of a couple of white dwarfs using *wdwarfdate*. First let's define the effective temperatures and surface gravity for both white dwarfs. 

```python
    import wdwarfdate
    import numpy as np

    #Define data for the white dwarfs
    teffs = np.array([19250,20250])
    teffs_err = np.array([500,850])
    loggs = np.array([8.16,8.526])
    loggs_err = np.array([0.084,0.126])

```

To run *wdwarfdate* we first set up the object WhiteDwarf with the parameters of the models we want to use. The code will sample the posterior distribution defined by the chosen models using [*emcee*](https://emcee.readthedocs.io/en/stable/) until convergence. 

```python
WD = wdwarfdate.WhiteDwarf(teffs,teffs_err,loggs,loggs_err,
                           model_wd='DA',feh='p0.00',vvcrit='0.0',
                           model_ifmr = 'Cummings_2018_MIST', 
                           high_perc = 84, low_perc = 16,
                           datatype='Gyr', save_plots=True, 
                           display_plots=True)
```

Then we run the parameter estimation.

```python
WD.calc_wd_age()
```
*wdwarfdate* allows us to select which models we want to use for the white dwarfs: the initial-to-final mass relation, DA or DB, and the parameter for the isochrone of the progenitor star, as shown above. Also with the *datatype* option we can select the units of the resulting ages. The results of the parameter estimation will be saved in an [*astropy* Table](https://docs.astropy.org/en/stable/table/index.html), which is saved in the object WD. To display the results we can do

```python
WD.results
```

This table will have one row for each teff and logg given and an estimation of the median of the distribution, the difference between the median and the 16th percentile of the distribution, and the difference between the 84th percentile and the median of the distribution for the following parameters:

- *ms_age:* Main sequence age, or age of the progenitor star.
- *cooling_age:* Cooling age of the white dwarf.
- *total_age:* Total age of the white dwarf, which is the sum of the main sequence and the cooling age.
- *initial_mass:* Initial mass, meaning the mass of the progenitor star.
- *final_mass:* Final mass, meaning the mass of the white dwarf.

When we run *wdwarfdate* it will also save five files per star in a folder called results (which the code will create if it doesn't exist):

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corner_plot.png which contains the corner plot for the three variables the code samples: main sequence age, cooling age and delta m.

2. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_walkers.png which contains the traces for each walker to confirm convergence of the code.

3. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age), and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age).

4. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corr_time.png which contains the result of the autocorrelation time as explained in the [tutorial](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/) by Dan Foreman-Mackey.

5. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST.txt which contains the likelihood evaluations in each step of the MCMC for all the parameters we are interested in. These are the columns: 'ln_ms_age', 'ln_cooling_age', 'ln_total_age, 'initial_mass' and 'final_mass'.

To find the initial condition, *wdwarfdate* runs the *fast_test* method. This method generates a gaussian distribution for each pair of *teff* and *logg* using the uncertainties as standard deviation, and pass the full distribution through a chain of models to calculate the total age of the white dwarfs, and the rest of the parameters described above. This provides a good first approximation to the parameters. This method is also available to run separaterly.

```python
WD = wdwarfdate.WhiteDwarf(teffs,teffs_err,loggs,loggs_err,
                           method='fast_test',
                           model_wd='DA',feh='p0.00',vvcrit='0.0',
                           model_ifmr = 'Cummings_2018_MIST',
                           high_perc = 84, low_perc = 16,
                           datatype='Gyr', return_distributions=True,
                           save_plots=True, display_plots=True)
WD.calc_wd_age()
```

The output of this method is one plot:

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_fast_test_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age) and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age), but obtained with the *fast_test* method described above.

For more explanation and examples checkout the [documentation](https://wdwarfdate.readthedocs.io/en/latest/) (under construction).


Citation
--------

If you use *wdwarfdate* in your reaserch, there will be a paper to cite: Kiman et al. in prep.

