import numpy as np
from astropy.table import Table
import os
import emcee
import corner
from .cooling_age import calc_cooling_age, get_cooling_model
from .ifmr import calc_initial_mass
from .ms_age import calc_ms_age, get_isochrone_model
from .extra_func import calc_percentiles, plot_distributions, check_ranges
from .bayesian_age import ln_posterior_prob
from .check_convergence import calc_auto_corr_time


class WhiteDwarf:
    def __init__(self, teff0, e_teff0, logg0, e_logg0, method, model_wd='DA',
                 feh='p0.00', vvcrit='0.0', model_ifmr='Cummings_2018_MIST',
                 init_params=[], high_perc=84, low_perc=16, datatype='yr',
                 path='results/', nburn_in=1000, max_n=100000,
                 n_indep_samples=100, n_mc=2000, return_distributions=False,
                 save_plot=False):
        """
        Parameters
        ----------
        teff0 : scalar, array. Effective temperature of the white dwarf
        e_teff0 : scalar, array. Error in the effective temperature of the white dwarf
        logg0 : scalar, array. Surface gravity of the white dwarf
        e_logg0 : scalar, arraya. Error in surface gravity of the white dwarf
        method : string. 'bayesian' or 'fast_test'. Bayesian will run an mcmc and
                 output the distributions. fast_test runs a normal distribution
                 centered at the value with a std of the error through all the
                 models chosen.
        model_wd : string. Spectral type of the white dwarf 'DA' or 'DB'.
        feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00','p0.00'
              or 'p0.50'
        vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'
        model_ifmr : string. Initial to final mass relation model. Can be
                     'Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                     'Salaris_2009' or 'Williams_2009'.
        init_params : list, array. Optional initial parameter for the burn in of
                      the mcmc for:
                      [log10 ms age, log10 cooling age, delta m].
                      Only useful in Bayesian mode.
        high_perc : scalar. Percentage at which the high errors will be calculated.
        low_perc : scalar. Percentage at which the low errors will be calculated.
        datatype : string. 'yr', 'Gyr' or 'log'. Units in which the results will be
                   output.
        path : string. Name of the folder where all the plots and distribution file
               will be save. If it doesn't exist, the code will create it.
        nburn_in : scalar. Number of steps for the burn in. Only useful in
                   Bayesian mode.
        max_n : scalar. Maximum number of steps done by the mcmc to estimate
                parameters. Only useful in Bayesian mode.
        n_indep_samples : scalar. Number of independent samples. The MCMC will run
                         for n_idep_samples*n_calc_auto_corr steps. Only useful in
                         Bayesian mode.
        n_mc : scalar. Length of the distribution for each parameter. Only
               useful in fast_test mode.
        return_distributions : True or False. Adds columns to the outputs with the
                               distributions of each parameter. Only useful in
                               fast_test mode.
        plot: True or Flase. If True, plots and saves the figures describing the
              result in the path given.
        """

        self.teff = teff0
        self.e_teff = e_teff0
        self.logg = logg0
        self.e_logg = e_logg0
        self.teff_i = 0
        self.e_teff_i = 0
        self.logg_i = 0
        self.e_logg_i = 0
        self.method = method
        self.model_wd = model_wd
        self.feh = feh
        self.vvcrit = vvcrit
        self.model_ifmr = model_ifmr
        self.init_params = init_params
        self.high_perc = high_perc
        self.low_perc = low_perc
        self.datatype = datatype
        self.path = path
        self.wd_path_id = ''
        self.nburn_in = nburn_in
        self.max_n = max_n
        self.n_indep_samples = n_indep_samples
        self.n_mc = n_mc
        self.models0 = []
        self.ndim = 3
        self.nwalkers = 70  # nwalkers > 2*ndim
        self.return_distributions = return_distributions
        self.save_plot = save_plot
        self.results = Table(
            names=('ms_age_median', 'ms_age_err_low', 'ms_age_err_high',
                   'cooling_age_median', 'cooling_age_err_low',
                   'cooling_age_err_high', 'total_age_median',
                   'total_age_err_low', 'total_age_err_high',
                   'initial_mass_median', 'initial_mass_err_low',
                   'initial_mass_err_high', 'final_mass_median',
                   'final_mass_err_low', 'final_mass_err_high'))

        if not isinstance(teff0, np.ndarray):
            self.teff = np.array([teff0])
            self.e_teff = np.array([e_teff0])
            self.logg = np.array([logg0])
            self.e_logg = np.array([e_logg0])

    def calc_wd_age(self):
        # If it doesn't exist, creates a folder to save the plots
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.method == 'bayesian':
            for x, y, z, w in zip(self.teff, self.e_teff, self.logg,
                                  self.e_logg):
                print(f'Running teff:{x} logg:{y}')
                check_ranges(x, y, self.model_wd)
                self.teff_i = x
                self.e_teff_i = y
                self.logg_i = z
                self.e_logg_i = w

                results_i = self.calc_bayesian_wd_age()

                self.results.add_row(results_i)

        elif self.method == 'fast_test':
            self.results = calc_wd_age_fast_test(teff0, e_teff0, logg0, e_logg0,
                                                 n_mc,
                                                 model_wd, feh, vvcrit,
                                                 model_ifmr,
                                                 high_perc, low_perc, datatype,
                                                 path,
                                                 return_distributions=return_distributions,
                                                 plot=plot)

    def calc_bayesian_wd_age(self):
        """
        Calculates percentiles for main sequence age, cooling age, total age,
        final mass and initial mass of a white dwarf with teff0 and logg0.
        Works for one white dwarf at a time.
        """

        # Set name of path and wd models to identif results
        self.wd_path_id = self.get_wd_path_id()

        # Interpolates models for cooling age and main sequence age
        cooling_models = get_cooling_model(self.model_wd)
        isochrone_model = get_isochrone_model(feh=self.feh, vvcrit=self.vvcrit)

        self.models0 = [self.model_ifmr, isochrone_model, cooling_models, '']

        if not self.init_params:
            self.init_params = self.get_initial_conditions()

        # Check if file exists and remove if it does so it can be filled again
        if os.path.exists(self.wd_path_id + '.txt'):
            os.remove(self.wd_path_id + '.txt')

        # Run emcee to obtain likelihood evaluations of ms age, cooling age,
        # total age, final mass and initial mass
        self.run_emcee()

        ln_ms_age = self.flat_samples[:, 0]
        ln_cooling_age = self.flat_samples[:, 1]

        # Open file where the likelihood evaluations where saved
        like_eval = np.loadtxt(self.wd_path_id + '.txt')

        # Use the likelihood evaluations for the dependent parameters
        # and the posterior for the independent parameters
        ln_total_age = like_eval[:, 2]
        initial_mass = like_eval[:, 3]
        final_mass = like_eval[:, 4]

        assert len(ln_ms_age) == len(ln_total_age)

        # Calculate percentiles for ms age, cooling age, total age,
        # initial mass and final mass
        res_ms_age = ln_ms_age
        res_cool_age = ln_cooling_age
        res_tot_age = ln_total_age
        if self.datatype == 'yr':
            res_ms_age = (10 ** ln_ms_age)
            res_cool_age = (10 ** ln_cooling_age)
            res_tot_age = (10 ** ln_total_age)
        elif self.datatype == 'Gyr':
            res_ms_age = (10 ** ln_ms_age) / 1e9
            res_cool_age = (10 ** ln_cooling_age) / 1e9
            res_tot_age = (10 ** ln_total_age) / 1e9

        results = calc_percentiles(res_ms_age, res_cool_age, res_tot_age,
                                   initial_mass, final_mass,
                                   self.high_perc, self.low_perc)

        if self.save_plot:
            # Plot corner plot with results from EMCEE for MS age, Total age,
            # and delta m.
            self.plot_results_mcmc(self.chain, self.ndim, self.wd_path_id)

            # Plot distribution for all the white dwarf and progenitor
            # parameters
            if datatype == 'yr':
                r_dummy = calc_percentiles(ln_ms_age, ln_cooling_age,
                                           ln_total_age, initial_mass,
                                           final_mass, self.high_perc,
                                           self.low_perc)
                plot_distributions(ln_ms_age, ln_cooling_age, ln_total_age,
                                   initial_mass, final_mass, self.datatype,
                                   r_dummy, name=self.wd_path_id)
            else:
                plot_distributions(res_ms_age, res_cool_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   results, name=self.wd_path_id)

            # Plot autocorrelation as a function of steps to confirm convergence
            self.plot_autocorr()
            
        return results

    def get_wd_path_id(self):
        # Set the name to identify the results from each white dwarf
        tg_name = 'teff_' + str(self.teff_i) + '_logg_' + str(self.logg_i)
        mist_name = '_feh_' + self.feh + '_vvcrit_' + self.vvcrit
        models_name = mist_name + '_' + self.model_wd + '_' + self.model_ifmr
        return self.path + tg_name + models_name

    def get_initial_conditions(self):
        """
        Runs fast-test method to obtain an approximate solution for the
        white dwarf parameters. These are used as initial conditions for the
        MCMC.

        Returns
        -------
        init_params: (array) with initial conditions for the MCMC.
        """

        if self.model_ifmr == 'Marigo_2020':
            ifmr_dummy = 'Cummings_2018_MIST'
        else:
            ifmr_dummy = self.model_ifmr

        teff_dist = np.array([[self.teff_i]])
        logg_dist = np.array([[self.logg_i]])
        cool_age_dist, final_mass_dist = calc_cooling_age(teff_dist, logg_dist,
                                                          1, self.model_wd)

        initial_mass_dist = calc_initial_mass(ifmr_dummy, final_mass_dist)
        ms_age_dist = calc_ms_age(initial_mass_dist, self.feh, self.vvcrit)

        init_params = np.array([np.log10(ms_age_dist[0][0]),
                                np.log10(cool_age_dist[0][0]), 0])

        if any(np.isnan(init_params)):
            teff_dist = np.random.normal(self.teff_i, self.e_teff_i, 1000)
            logg_dist = np.random.normal(self.logg_i, self.e_logg_i, 1000)
            cool_age_dist, final_mass_dist = calc_cooling_age(teff_dist,
                                                              logg_dist,
                                                              1, self.model_wd)
            initial_mass_dist = calc_initial_mass(ifmr_dummy, final_mass_dist)
            ms_age_dist = calc_ms_age(initial_mass_dist, self.feh, self.vvcrit)

            init_params = np.array([np.nanmedian(np.log10(ms_age_dist[0])),
                                    np.nanmedian(np.log10(cool_age_dist[0])),
                                    0])

        return init_params

    def run_emcee(self):
        """
        Starting from the maximum likelihood ages (main sequence age and cooling
        age), samples the posterior to get the likelihood evaluations of the
        rest of the parameters (final mass, initial mass and total age)
        models0 : list. [model_ifmr,isochrone_model,cooling_models,wd_path_id]
        """

        # Initialize walkers
        p0 = np.array([self.init_params
                       + np.random.uniform(-.05, .05, 3) for i in
                       range(nwalkers)])

        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                        ln_posterior_prob,
                                        args=[self.teff_i, self.e_teff_i,
                                              self.logg_i, self.e_logg_i,
                                              self.models0])

        # Run burn in
        p0_new, _, _ = sampler.run_mcmc(p0, self.nburn_in)

        save_likelihoods_file = self.wd_path_id + '.txt'
        with open(save_likelihoods_file, 'a') as f:
            # Set names of file columns
            f.write('#ln_ms_age ln_cooling_age ln_total_age initial_mass '
                    'final_mass\n')
            self.models0[3] = f

            # Initialize sampler again but now so it saves likelihood
            # evaluations
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                            ln_posterior_prob,
                                            args=[self.teff_i, self.e_teff_i,
                                                  self.logg_i, self.e_logg_i,
                                                  self.models0])
            n_steps = int(self.max_n / 100)

            index = 0
            autocorr = np.empty(self.max_n)

            # This will be useful to testing convergence
            old_tau = np.inf

            # Run MCMC checking convergence every 100 steps
            for x in range(n_steps):
                if x % 100 == 0:
                    print(f"{x} steps out of {n_steps}")
                p0_new, _, _ = sampler.run_mcmc(p0_new, 100)
                chain = sampler.chain
                # Compute the autocorrelation time so far
                tau = calc_auto_corr_time(chain)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * n_indep_samples < (x + 1) * 100)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    print('Converged')
                    self.index_conv_i = index
                    self.autocorr_i = autocorr
                    break
                old_tau = tau

            # Obtain chain of samples
            self.chain = sampler.chain[:, :, :]
            self.flat_samples = chain.reshape((-1, self.ndim))

        return 0

    def plot_autocorr(self):
        N = 100 * np.arange(1, self.index_conv_i + 1)
        plt.plot(N, N / 100.0, "--k", label=r"$\tau = N/100$")
        plt.loglog(N, self.autocorr_i[:self.index_conv_i], "-")
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"mean $\hat{\tau}$")
        plt.legend(fontsize=14)
        plt.grid()
        plt.savefig(self.wd_path_id + '_corr_time.png')
        plt.close()

    def plot_results_mcmc(self):

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        for i in range(50):
            ax1.plot(self.chain[i, :, 0], color='k', alpha=0.05)
            ax1.axhline(y=np.median(self.flat_samples[:, 0]), color='k')
        ax1.set_ylabel(r'$\log_{10}(t_{\rm ms}/{\rm yr})$')

        for i in range(50):
            ax2.plot(self.chain[i, :, 1], color='k', alpha=0.05)
            ax2.axhline(y=np.median(self.flat_samples[:, 1]), color='k')
        ax2.set_ylabel(r'$\log_{10}(t_{\rm cool}/{\rm yr})$')

        for i in range(50):
            ax3.plot(self.chain[i, :, 2], color='k', alpha=0.05)
            ax3.axhline(y=np.median(self.flat_samples[:, 2]), color='k')
        ax3.set_ylabel(r'$\Delta_{\rm m}$')
        plt.tight_layout()
        plt.savefig(wd_path_id + '_walkers.png')
        plt.close(f)

        labels = [r'$\log_{10}(t_{\rm ms}/{\rm yr})$',
                  r'$\log_{10}(t_{\rm cool}/{\rm yr})$',
                  r'$\Delta _{\rm m}$']

        fig = corner.corner(self.flat_samples, labels=labels,
                            quantiles=[.16, .50, .84],
                            show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig(self.d_path_id + '_corner_plot.png', dpi=300)
        plt.close(fig)


def calc_wd_age_fast_test(teff0, e_teff0, logg0, e_logg0, n_mc, model_wd, feh,
                          vvcrit,
                          model_ifmr, high_perc, low_perc, datatype,
                          path,
                          return_distributions, plot):
    """
    Calculated white dwarfs ages with a fast_test approach. Starts from normal
    distribution of teff and logg based on the errors and passes the full
    distribution through the same process to get a distribution of ages.
    """

    if not isinstance(teff0, np.ndarray):
        teff0 = np.array([teff0])
        e_teff0 = np.array([e_teff0])
        logg0 = np.array([logg0])
        e_logg0 = np.array([e_logg0])

    N = len(teff0)

    # Set up the distribution of teff and logg
    teff_dist, logg_dist = [], []
    for i in range(N):
        if np.isnan(teff0[i] + e_teff0[i] + logg0[i] + e_logg0[i]):
            teff_dist.append(np.nan)
            logg_dist.append(np.nan)
        else:
            check_ranges(teff0[i], logg0[i], model_wd)
            teff_dist.append(np.random.normal(teff0[i], e_teff0[i], n_mc))
            logg_dist.append(np.random.normal(logg0[i], e_logg0[i], n_mc))
    teff_dist, logg_dist = np.array(teff_dist), np.array(logg_dist)

    # From teff and logg get ages
    cooling_age_dist, final_mass_dist = calc_cooling_age(teff_dist, logg_dist,
                                                         N, model=model_wd)
    initial_mass_dist = calc_initial_mass(model_ifmr, final_mass_dist)
    ms_age_dist = calc_ms_age(initial_mass_dist, feh=feh, vvcrit=vvcrit)
    total_age_dist = cooling_age_dist + ms_age_dist

    # Replace all the ages which are higher than the age of the universe with
    # nans
    mask_nan = np.isnan(total_age_dist)
    total_age_dist[mask_nan] = -1

    mask = total_age_dist / 1e9 > 13.8
    total_age_dist[mask] = np.nan

    total_age_dist[mask_nan] = np.nan

    cooling_age_dist[mask] = np.copy(cooling_age_dist[mask]) * np.nan
    final_mass_dist[mask] = np.copy(final_mass_dist[mask]) * np.nan
    initial_mass_dist[mask] = np.copy(initial_mass_dist[mask]) * np.nan
    ms_age_dist[mask] = np.copy(ms_age_dist[mask]) * np.nan
    total_age_dist[mask] = np.copy(total_age_dist[mask]) * np.nan

    # Calculate percentiles and save results
    results = Table()

    median, high_err, low_err = calc_dist_percentiles(final_mass_dist, 'none',
                                                      high_perc, low_perc)
    results['final_mass_median'] = median
    results['final_mass_err_high'] = high_err
    results['final_mass_err_low'] = low_err

    median, high_err, low_err = calc_dist_percentiles(initial_mass_dist, 'none',
                                                      high_perc, low_perc)
    results['initial_mass_median'] = median
    results['initial_mass_err_high'] = high_err
    results['initial_mass_err_low'] = low_err

    median, high_err, low_err = calc_dist_percentiles(cooling_age_dist,
                                                      datatype,
                                                      high_perc, low_perc)
    results['cooling_age_median'] = median
    results['cooling_age_err_high'] = high_err
    results['cooling_age_err_low'] = low_err

    median, high_err, low_err = calc_dist_percentiles(ms_age_dist, datatype,
                                                      high_perc, low_perc)
    results['ms_age_median'] = median
    results['ms_age_err_high'] = high_err
    results['ms_age_err_low'] = low_err

    median, high_err, low_err = calc_dist_percentiles(total_age_dist, datatype,
                                                      high_perc, low_perc)
    results['total_age_median'] = median
    results['total_age_err_high'] = high_err
    results['total_age_err_low'] = low_err

    if return_distributions:
        results['final_mass_dist'] = final_mass_dist
        results['initial_mass_dist'] = initial_mass_dist
        if datatype == 'yr':
            results['cooling_age_dist'] = cooling_age_dist
            results['ms_age_dist'] = ms_age_dist
            results['total_age_dist'] = total_age_dist
        elif datatype == 'Gyr':
            results['cooling_age_dist'] = cooling_age_dist / 1e9
            results['ms_age_dist'] = ms_age_dist / 1e9
            results['total_age_dist'] = total_age_dist / 1e9
        elif datatype == 'log':
            results['cooling_age_dist'] = np.log10(cooling_age_dist)
            results['ms_age_dist'] = np.log10(ms_age_dist)
            results['total_age_dist'] = np.log10(total_age_dist)

    # Plot resulting distributions
    if plot == True:
        if not os.path.exists(path):
            os.makedirs(path)

        for x1, x2, x3, x4, x5, x6, x7 in zip(teff0, logg0, ms_age_dist,
                                              cooling_age_dist,
                                              total_age_dist,
                                              initial_mass_dist,
                                              final_mass_dist):
            wd_path_id = get_wd_path_id(x1, x2, feh, vvcrit, model_wd,
                                        model_ifmr, path)
            x3 = np.array(x3)
            x4 = np.array(x4)
            x5 = np.array(x5)
            x6 = np.array(x6)
            x7 = np.array(x7)

            if datatype == 'yr':
                plot_distributions(x3, x4, x5,
                                   x6, x7, high_perc=high_perc,
                                   low_perc=low_perc,
                                   datatype=datatype,
                                   name=wd_path_id + '_fast_test')
            elif datatype == 'Gyr':
                plot_distributions(x3 / 1e9, x4 / 1e9, x5 / 1e9,
                                   x6, x7, high_perc, low_perc, datatype,
                                   name=wd_path_id + '_fast_test')
            elif datatype == 'log':
                plot_distributions(np.log10(x3), np.log10(x4),
                                   np.log10(x5),
                                   x6, x7, high_perc, low_perc, datatype,
                                   name=wd_path_id + '_fast_test')

    return results


def calc_dist_percentiles(dist, datatype, high_perc, low_perc):
    if datatype == 'yr' or datatype == 'none':
        median = np.array([np.nanpercentile(x, 50) for x in dist])
        h = [np.nanpercentile(x, high_perc) - np.nanpercentile(x, 50) for x in
             dist]
        l = [np.nanpercentile(x, 50) - np.nanpercentile(x, low_perc) for x in
             dist]
        h, l = np.array(h), np.array(l)
    elif datatype == 'Gyr':
        dist1 = dist / 1e9
        median = np.array([np.nanpercentile(x, 50) for x in dist1])
        h = [np.nanpercentile(x, high_perc) - np.nanpercentile(x, 50) for x in
             dist1]
        l = [np.nanpercentile(x, 50) - np.nanpercentile(x, low_perc) for x in
             dist1]
        h, l = np.array(h), np.array(l)
    elif datatype == 'log':
        dist1 = np.log10(dist)
        median = np.array([np.nanpercentile(x, 50) for x in dist1])
        h = [np.nanpercentile(x, high_perc) - np.nanpercentile(x, 50) for x in
             dist1]
        l = [np.nanpercentile(x, 50) - np.nanpercentile(x, low_perc) for x in
             dist1]
        h, l = np.array(h), np.array(l)
    return median, h, l
