
wdwarfdate
====================================

*wdwarfdate* is a code which combines different models in a bayesian framework to calculate ages of white dwarfs from an effective temperature and a surface gravity. 



Installation
============

```bash
    git clone https://github.com/rkiman/wdwarfdate.git
    cd wdwarfdate
    python setup.py install
```

Example usage
-------------
```python
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
```

To run *wdwarfdate* we just call the function that calculates ages and choose which method we want to use: 'bayesian' of 'freq'. The 'freq' method is much more faster so it is recommended to have a first approximation of the ages. The 'bayesian' method produces a better estimation of the errors.

```python
data_comparison = [[t_ms[i],t_cool[i],t_tot[i],m_i[i],m_f[i]] for i in range(len(teff))]
results = wdwarfdate.calc_wd_age(teff,teff_err,logg,logg_err,method='bayesian',
                                                        comparison = data_comparison,
                                                        nburn_in = 100,n_calc_auto_corr = 100,
                                                        n_indep_samples = 100)
```

*wdwarfdate* allows you to select which models you want to use for the white dwarfs: the initial-final mass relation, DA or DB, and the parameter for the isochrone. 
This run will save three files per star in a folder called results:

1. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_corner_plot.png which contains the corner plot for the three variables the code samples: main sequence age, cooling age and delta m.

2. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_walkers.png which contains the traces for each walker to confirm convergence of the code.

3. teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_distributions.png which contains the distribution of all the parameter of the white dwarf: the sampled parameters (main sequence and cooling age) and the likelihood evaluations for the dependent parameters (final mass, initial mass and total age).

The variable *results* now contains the percentiles indicated and the median for each parameter.

```python
    ms_age_median = results[:,0]
    ms_age_err_low = results[:,1]
    ms_age_err_high = results[:,2]
    cooling_age_median = results[:,3]
    cooling_age_err_low = results[:,4]
    cooling_age_err_high = results[:,5]
    total_age_median = results[:,6]
    total_age_err_low = results[:,7]
    total_age_err_high = results[:,8]
    initial_mass_median = results[:,9]
    initial_mass_err_low = results[:,10]
    initial_mass_err_high = results[:,11]
    final_mass_median = results[:,12]
    final_mass_err_low = results[:,13]
    final_mass_err_high = results[:,14]
```

Coming soon: Documentation
