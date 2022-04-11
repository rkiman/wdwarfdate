
wdwarfdate
==========

`wdwarfdate` is a Python open source code which derives the total age of a white dwarf from an effective temperature and a surface gravity in a Bayesian framework. *wdwarfdate* runs a chain of models assuming single star evolution and estimate the following parameters and their uncertainties: total age of the object, mass and cooling age of the white dwarf and mass and lifetime of the progenitor star. Checkout the documentation for *wdwarfdate* [here](https://wdwarfdate.readthedocs.io/en/latest/) (under construction).


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

Getting right to it, let's make and example on how to estimate the ages of a couple of white dwarfs using *wdwarfdate*. First let's define the effective temperatures and surface gravity for both white dwarfs ([Cummings et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...866...21C/abstract)).

```python
import wdwarfdate    # Import the code wdwarfdate
import numpy as np

#Define data for the white dwarfs
teffs = np.array([19250,20250])
teffs_err = np.array([500,850])
loggs = np.array([8.16,8.526])
loggs_err = np.array([0.084,0.126])
```

To run *wdwarfdate* we first set up the object WhiteDwarf with the parameters of the models we want to use. The code will sample the posterior distribution defined by the chosen models using the grid method. The parameters which are being sampled are the mass of the progenitor star, the cooling age of the white dwarf and a &#916;m parameter to model the scatter in the initial-to-final mass relation. The limits of the grid will be set automatically but can be set manually in this step too using `min_mi` and `max_mi` for the mass, and `min_log10_tcool` and `max_log10_tcool` for the cooling age. 

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
*wdwarfdate* allows us to select which models we want to use for the white dwarfs: the initial-to-final mass relation, DA or non-DA, and the parameter for the isochrone of the progenitor star, as shown above. Also with the *datatype* option we can select the units of the resulting ages. The results of the parameter estimation will be saved in an [*astropy* Table](https://docs.astropy.org/en/stable/table/index.html), which is saved in the object WD. To display the results we can do

```python
WD.results
```

This table will have one row for each teff and logg given and an estimation of the median of the distribution, the difference between the median and the 16th percentile of the distribution, and the difference between the 84th percentile and the median of the distribution for the following parameters:

- *ms_age:* Main sequence age, or the lifetime of progenitor star.
- *cooling_age:* Cooling age of the white dwarf.
- *total_age:* Total age of the white dwarf, which is the sum of the main sequence and the cooling age.
- *initial_mass:* Initial mass, meaning the mass of the progenitor star.
- *final_mass:* Final mass, meaning the mass of the white dwarf.

When we run *wdwarfdate* we can set `display_plots=True` to display the grid plot with the parameters which were sampled with the code and an extra plot with the distributions of all the parameters, as shown below for the first white dwarf selected above. If we set `save_plots=True`, these plots will be saved in a folder called results, unless another path is indicated. 

![Grid plot](https://github.com/rkiman/wdwarfdate/blob/master/docs/source/tutorials/results/teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_gridplot.png)

![Distributions plot](https://github.com/rkiman/wdwarfdate/blob/master/docs/source/tutorials/results/teff_19250_logg_8.16_feh_p0.00_vvcrit_0.0_DA_Cummings_2018_MIST_distributions.png)


For more explanation and examples checkout the [documentation](https://wdwarfdate.readthedocs.io/en/latest/) (under construction).


Citation
--------

If you use *wdwarfdate* in your reaserch, there will be a paper to cite: Kiman et al. in prep.

