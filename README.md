
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

To run *wdwarfdate* we need call the function that calculates ages and choose which method we want to use: *bayesian* of *fast_test*, as shown below. The *bayesian* method will run a Markov Chain Monte Carlo using [*emcee*](https://emcee.readthedocs.io/en/stable/) until convergence. The *fast_test* method will generate a gaussian distribution for *teff* and *logg* using the uncertainties as starndard deviation, and pass the full distribution through a chain of models to calculate the total age of the white dwarfs and other parameters described below. To calculate the ages of the two white dwarfs we initiated above using the *bayesian* method we can do something like

```python
results_bayes = wdwarfdate.calc_wd_age(teffs,teffs_err,loggs,loggs_err,
                                       method='bayesian', datatype='Gyr')
```

The *bayesian* method produces a better estimation of the uncertainties than the *fast_test*. However, the *bayesian* method will take a long time to run because it is going to run the MCMC until convergence. The *fast_test* method is faster so it is recommended to have a first approximation of the ages and their uncertainties. To calculate the ages of white dwarfs in the *fast_test* mode we can do

```python
results_fast_test = wdwarfdate.calc_wd_age(teffs,teffs_err,loggs,loggs_err,
                                           method='fast_test', datatype='Gyr')
```

*wdwarfdate* allows us to select which models we want to use for the white dwarfs: the initial-to-final mass relation, DA or DB, and the parameter for the isochrone. As we did not specified the models in this example, the outputs are going to be estimated assuming these are DA white dwarfs, using the Cummings et al. 2018 MIST initial-to-final mass relation and assuming solar metallicity for the progenitor star. Also with the *datatype* option we can select the units of the resulting ages. 

The output of this function will be an astropy Table with one row for each teff and logg given and the following columns:

- *ms_age_median:* Median values of main sequence age distribution of the progenitor of the white dwarf
- *ms_age_err_low:* The difference between the median and the 16th percentile of the main sequence age distribution
- *ms_age_err_high:* The difference between the 84th percentile and the median of the main sequence age distribution
- *cooling_age_median:* Median values of cooling age distribution of the white dwarf
- *cooling_age_err_low:* The difference between the median and the 16th percentile of the cooling age distribution
- *cooling_age_err_high:* The difference between the 84th percentile and the median of the cooling age distribution
- *total_age_median:* Median values of total age distribution of the white dwarf
- *total_age_err_low:* The difference between the median and the 16th percentile of the total age distribution
- *total_age_err_high:* The difference between the 84th percentile and the median of the total age distribution
- *initial_mass_median:* Median values of initial mass distribution, meaning the mass of the progenitor
- *initial_mass_err_low:* The difference between the median and the 16th percentile of the initial mass distribution
- *initial_mass_err_high:* The difference between the 84th percentile and the median of the initial mass distribution
- *final_mass_median:* Median values of final mass distribution, meaning the mass of the white dwarf 
- *final_mass_err_low:* The difference between the median and the 16th percentile of the final mass distribution
- *final_mass_err_high:* The difference between the 84th percentile and the median of the final mass distribution

When we run the *bayesian* method, *wdwarfdate* will also save five files per star in a folder called results (which the code will create if it doesn't exist):

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corner_plot.png which contains the corner plot for the three variables the code samples: main sequence age, cooling age and delta m.

2. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_walkers.png which contains the traces for each walker to confirm convergence of the code.

3. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age) and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age).

4. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corr_time.png which contains the result of the autocorrelation time as explained in the [tutorial](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/) by Dan Foreman-Mackey.

5. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST.txt which contains the likelihood evaluations in each step of the MCMC for all the parameters we are interested in. These are the columns: 'ln_ms_age', 'ln_cooling_age', 'ln_total_age, 'initial_mass' and 'final_mass'.

When we run the *fast_test* method, *wdwarfdate* will save one file:

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_fast_test_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age) and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age), but obtained with the *fast_test* method described above.

For more explanation and examples checkout the [documentation](https://wdwarfdate.readthedocs.io/en/latest/) (under construction).


Citation
--------

If you use *wdwarfdate* in your reaserch, there will be a paper to cite: Kiman et al. in prep.

