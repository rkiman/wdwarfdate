import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import os
import emcee
import corner
import warnings
import time
from .cooling_age import calc_cooling_age, get_cooling_model
from .ifmr import calc_initial_mass, ifmr_bayesian
from .ms_age import calc_ms_age, get_isochrone_model
from .extra_func import calc_percentiles, plot_distributions
from .extra_func import calc_dist_percentiles
from .bayesian_age import ln_posterior_prob
from .check_convergence import calc_auto_corr_time


class WhiteDwarf:
    def __init__(self, teff0, e_teff0, logg0, e_logg0, method='bayesian',
                 model_wd='DA', feh='p0.00', vvcrit='0.0',
                 model_ifmr='Cummings_2018_MIST', init_params=[],
                 high_perc=84,
                 low_perc=16, datatype='yr', path='results/', nburn_in=2,
                 max_n=100000, n_indep_samples=100, n_mc=2000,
                 return_distributions=False, save_plots=False,
                 display_plots=True, save_log=False):
        """
        Parameters
        ----------
        teff0 : scalar, array. Effective temperature of the white dwarf
        e_teff0 : scalar, array. Error in the effective temperature of the white
        dwarf
        logg0 : scalar, array. Surface gravity of the white dwarf
        e_logg0 : scalar, arraya. Error in surface gravity of the white dwarf
        method : string. 'bayesian' or 'fast_test'. Bayesian will run an mcmc
                 and output the distributions. fast_test runs a normal
                 distribution centered at the value with a std of the error
                 through all the models chosen.
        model_wd : string. Spectral type of the white dwarf 'DA' or 'DB'.
        feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00',
              'p0.00' or 'p0.50'
        vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'
        model_ifmr : string. Initial to final mass relation model. Can be
                     'Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                     'Salaris_2009' or 'Williams_2009'.
        init_params : list, array. Optional initial parameter for the burn in of
                      the mcmc for:
                      [log10 ms age, log10 cooling age, delta m].
                      Only useful in Bayesian mode.
        high_perc : scalar. Percentage at which the high errors will be
                    calculated.
        low_perc : scalar. Percentage at which the low errors will be
                   calculated.
        datatype : string. 'yr', 'Gyr' or 'log'. Units in which the results
                   will be output.
        path : string. Name of the folder where all the plots and distribution
               file will be save. If it doesn't exist, the code will create it.
        nburn_in : scalar. Number of steps for the burn in. Only useful in
                   Bayesian mode.
        max_n : scalar. Maximum number of steps done by the mcmc to estimate
                parameters. Only useful in Bayesian mode.
        n_indep_samples : scalar. Number of independent samples. The MCMC will
                          run for n_idep_samples*n_calc_auto_corr steps. Only
                          useful in Bayesian mode.
        n_mc : scalar. Length of the distribution for each parameter. Only
               useful in fast_test mode.
        return_distributions : True or False. Adds columns to the outputs with
                               the distributions of each parameter. Only useful
                               in fast_test mode.
        save_plots: True or Flase. If True, plots and saves the figures
                   describing the result in the path given.
        """

        self.teff = teff0
        self.e_teff = e_teff0
        self.logg = logg0
        self.e_logg = e_logg0
        if not isinstance(teff0, np.ndarray):
            self.teff = np.array([teff0])
            self.e_teff = np.array([e_teff0])
            self.logg = np.array([logg0])
            self.e_logg = np.array([e_logg0])
        self.teff_i = 0
        self.e_teff_i = 0
        self.logg_i = 0
        self.e_logg_i = 0
        self.method = method
        self.model_wd = model_wd
        self.feh = feh
        self.vvcrit = vvcrit
        self.model_ifmr = model_ifmr
        self.n = len(self.teff)
        self.high_perc = high_perc
        self.low_perc = low_perc
        self.datatype = datatype
        self.path = path
        self.wd_path_id = ''
        self.save_plots = save_plots
        self.display_plots = display_plots
        self.save_log = save_log
        self.n_mc = n_mc
        # Bayesian method objects.
        if self.method == 'bayesian':
            if not init_params:
                if self.n > 1:
                    self.init_params = [[] for i in range(self.n)]
                else:
                    self.init_params = [[]]
            else:
                self.init_params = init_params
            self.init_params_i = []
            self.nburn_in = nburn_in
            self.max_n = max_n
            self.n_indep_samples = n_indep_samples
            self.models0 = []
            self.ndim = 3
            self.nwalkers = 70  # nwalkers > 2*ndim
            self.results = Table(
                names=('ms_age_median', 'ms_age_err_low', 'ms_age_err_high',
                       'cooling_age_median', 'cooling_age_err_low',
                       'cooling_age_err_high', 'total_age_median',
                       'total_age_err_low', 'total_age_err_high',
                       'initial_mass_median', 'initial_mass_err_low',
                       'initial_mass_err_high', 'final_mass_median',
                       'final_mass_err_low', 'final_mass_err_high'))

        # Fast-test method objects.
        if self.method == 'fast_test':
            self.return_distributions = return_distributions
            self.results_fast_test = Table(
                names=('ms_age_median', 'ms_age_err_low', 'ms_age_err_high',
                       'cooling_age_median', 'cooling_age_err_low',
                       'cooling_age_err_high', 'total_age_median',
                       'total_age_err_low', 'total_age_err_high',
                       'initial_mass_median', 'initial_mass_err_low',
                       'initial_mass_err_high', 'final_mass_median',
                       'final_mass_err_low', 'final_mass_err_high'))

    def calc_wd_age(self):
        # If it doesn't exist, creates a folder to save results.
        if self.save_plots:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        if self.method == 'bayesian':
            if self.save_log:
                file_log = open(self.path + 'run_log.txt', 'a')
                file_log.write('#Teff\tlogg\ttime (min)\tconverged\n')
            for x, y, z, w, q in zip(self.teff, self.e_teff, self.logg,
                                     self.e_logg, self.init_params):
                print(f'Running Teff:{x} logg:{z}')
                start = time.time()
                self.teff_i = x
                self.e_teff_i = y
                self.logg_i = z
                self.e_logg_i = w
                self.init_params_i = q
                self.converged = False

                # Set name of path and wd models to identify results
                self.wd_path_id = self.get_wd_path_id()

                results_i = self.calc_wd_age_bayesian()

                self.results.add_row(results_i)
                end = time.time()
                if self.save_log:
                    file_log.write(str(x) + '\t' + str(z) + '\t' +
                                   str((end - start) / 60) + '\t' +
                                   str('Y' if self.converged else 'N') + '\n')

        elif self.method == 'fast_test':
            self.check_teff_logg()
            self.calc_wd_age_fast_test()

    def calc_wd_age_bayesian(self):

        r = self.calc_single_star_params()
        cool_age, final_mass, initial_mass, ms_age = r

        if np.isnan(cool_age + final_mass) or cool_age < np.log10(3e5):
            text = "Effective temperature and/or surface " \
                   "gravity are outside of the allowed values of " \
                   "the models. Teff: %0.2f, logg: %1.2f" % (self.teff_i, self.logg_i)
            warnings.warn(text)
            results_i = np.ones(15) * np.nan
        elif ~np.isnan(final_mass) and np.isnan(initial_mass):
            text = "Final mass " \
                   "is outside the range allowed " \
                   "by the IFMR. Cannot estimate initial mass, main " \
                   "sequence age or total age. Final mass and " \
                   "cooling age were calculated with the fast_test " \
                   "method. " \
                   "Teff: %0.2f, logg: %1.2f, Final mass ~ %2.2f Msun " % (self.teff_i, self.logg_i, final_mass)
            warnings.warn(text)
            # Calculate final mass and cooling age with fast_test method
            results_i = self.calc_final_mass_cooling_age()
        elif (~np.isnan(final_mass + cool_age + initial_mass)
              and np.isnan(ms_age)):
            text = "Initial mass " \
                   "is outside of the range " \
                   "allowed by the MIST isochrones. Cannot estimate " \
                   "main sequence age or total age. Run the fast_test " \
                   "method to obtain a result for the rest of the " \
                   "parameters. " \
                   "Teff: %0.2f, logg: %1.2f, Initial mass ~ %2.2f Msun " % (self.teff_i, self.logg_i, initial_mass)
            warnings.warn(text)
            results_i = self.calc_final_mass_cooling_age()
        else:
            if final_mass < 0.56 or final_mass > 1.2414:
                text = "The IFMR is going to be extrapolated to " \
                       "calculate initial mass, main sequence " \
                       "age and total age. Use these parameters with " \
                       "caution. " \
                       "Teff: %0.2f, logg: %1.2f, Final mass ~ %2.2f Msun " % (self.teff_i, self.logg_i, final_mass)
                warnings.warn(text)
            results_i = self.run_calc_bayesian_wd_age()

        return results_i

    def get_wd_path_id(self):
        # Set the name to identify the results from each white dwarf
        tg_name = 'teff_' + str(self.teff_i) + '_logg_' + str(self.logg_i)
        mist_name = '_feh_' + self.feh + '_vvcrit_' + self.vvcrit
        models_name = mist_name + '_' + self.model_wd + '_' + self.model_ifmr
        return self.path + tg_name + models_name

    def calc_single_star_params(self):

        if self.model_ifmr == 'Marigo_2020':
            ifmr_dummy = 'Cummings_2018_MIST'
        else:
            ifmr_dummy = self.model_ifmr

        # Trick to make the following functions work.
        teff_dist = [np.ones(2)*self.teff_i, np.ones(2)*self.teff_i]
        logg_dist = [np.ones(2)*self.logg_i, np.ones(2)*self.logg_i]

        cool_age_dist, final_mass_dist = calc_cooling_age(teff_dist, logg_dist,
                                                          self.model_wd)

        initial_mass_dist = calc_initial_mass(ifmr_dummy, final_mass_dist)

        ms_age_dist = calc_ms_age(initial_mass_dist, self.feh, self.vvcrit)

        return [cool_age_dist[0][0], final_mass_dist[0][0],
                initial_mass_dist[0][0], ms_age_dist[0][0]]

    def run_calc_bayesian_wd_age(self):
        """
        Calculates percentiles for main sequence age, cooling age, total age,
        final mass and initial mass of a white dwarf with teff0 and logg0.
        Works for one white dwarf at a time.
        """

        # Interpolates models for cooling age and main sequence age
        self.cooling_models = get_cooling_model(self.model_wd)
        self.isochrone_model = get_isochrone_model(feh=self.feh,
                                                   vvcrit=self.vvcrit)

        self.models0 = [self.model_ifmr, self.isochrone_model,
                        self.cooling_models]

        if not self.init_params_i:
            self.init_params_i = self.get_initial_conditions()

        # Run emcee to obtain likelihood evaluations of ms age, cooling age,
        # total age, final mass and initial mass
        self.run_emcee()

        results_i = self.calculate_and_plot_results_bayesian()

        return np.array(results_i)

    def get_initial_conditions(self):
        """
        Runs fast-test method to obtain an approximate solution for the
        white dwarf parameters. These are used as initial conditions for the
        MCMC.

        Returns
        -------
        init_params: (array) with initial conditions for the MCMC.
        """

        cool_age_j, _, _, ms_age_j = self.calc_single_star_params()

        init_params = np.array([ms_age_j, cool_age_j, 0])

        if any(np.isnan(init_params)):
            teff_dist = [np.random.normal(self.teff_i, self.e_teff_i, self.n_mc)]
            logg_dist = [np.random.normal(self.logg_i, self.e_logg_i, self.n_mc)]
            cool_age_dist, final_mass_dist = calc_cooling_age(teff_dist,
                                                              logg_dist,
                                                              self.model_wd)
            if self.model_ifmr == 'Marigo_2020':
                ifmr_dummy = 'Cummings_2018_MIST'
            else:
                ifmr_dummy = self.model_ifmr

            initial_mass_dist = calc_initial_mass(ifmr_dummy, final_mass_dist)

            ms_age_dist = calc_ms_age(initial_mass_dist, self.feh, self.vvcrit)

            init_params = np.array([np.nanmedian(ms_age_dist[0]),
                                    np.nanmedian(cool_age_dist[0]),
                                    0])

        return init_params

    def calc_final_mass_cooling_age(self):
        """
        This function estimates final mass and cooling age in the Bayesian
        method, in the case that the initial mass, main sequence age and
        total age cannot be estimated because the final mass is outside
        the calibrated regime of the IFMR.

        """
        # Estimate final mass and cooling age distributions using a
        # Monte Carlo propagation of uncertainties.
        teff_dist = [np.random.normal(self.teff_i, self.e_teff_i, self.n_mc)]
        logg_dist = [np.random.normal(self.logg_i, self.e_logg_i, self.n_mc)]
        ln_cooling_age, final_mass = calc_cooling_age(teff_dist, logg_dist,
                                                      self.model_wd)
        ln_cooling_age = ln_cooling_age[0]
        final_mass = final_mass[0]

        # The rest of the parameters cannot be estimated, so the array is
        # filled with nans.
        res_ms_age = np.ones(self.n_mc)*np.nan
        res_tot_age = np.ones(self.n_mc)*np.nan
        initial_mass = np.ones(self.n_mc)*np.nan

        # Adjust units for the ages as required by the user.
        if self.datatype == 'yr':
            res_cool_age = (10 ** ln_cooling_age)
        elif self.datatype == 'Gyr':
            res_cool_age = (10 ** ln_cooling_age) / 1e9
        else:
            res_cool_age = ln_cooling_age

        # Estimate medians and uncertainties for each parameter.
        results_i = calc_percentiles(res_ms_age, res_cool_age, res_tot_age,
                                     initial_mass, final_mass,
                                     self.high_perc, self.low_perc)

        if self.display_plots or self.save_plots:

            # Plot distribution for all the white dwarf and progenitor
            # parameters
            if self.datatype == 'yr':
                r_dummy = calc_percentiles(res_ms_age, ln_cooling_age,
                                           res_tot_age, initial_mass,
                                           final_mass, self.high_perc,
                                           self.low_perc)
                plot_distributions(res_ms_age, ln_cooling_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   r_dummy, self.display_plots, self.save_plots,
                                   name=self.wd_path_id)
            else:
                plot_distributions(res_ms_age, res_cool_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   results_i, self.display_plots,
                                   self.save_plots, name=self.wd_path_id)

        # Save distributions of each parameter.
        file = open(self.wd_path_id + "_distributions.txt", 'a')
        file.write('#ms_age\tcool_age\ttotal_age\tinitial_mass\tfinal_mass\n')
        for x1, x2, x3, x4, x5 in zip(res_ms_age, res_cool_age, res_tot_age,
                                      initial_mass, final_mass):
            file.write(str(x1) + '\t' + str(x2) + '\t' + str(x3) + '\t' +
                       str(x4) + '\t' + str(x5) + '\n')
        file.close()

        return results_i

    def run_emcee(self):
        """
        Starting from the maximum likelihood ages (main sequence age and cooling
        age), samples the posterior to get the likelihood evaluations of the
        rest of the parameters (final mass, initial mass and total age)
        models0 : list. [model_ifmr,isochrone_model,cooling_models,wd_path_id]
        """

        # Initialize walkers
        p0 = np.array([self.init_params_i
                       + np.random.uniform(-.05, .05, 3) for i in
                       range(self.nwalkers)])

        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                        ln_posterior_prob,
                                        args=[self.teff_i, self.e_teff_i,
                                              self.logg_i, self.e_logg_i,
                                              self.models0])

        # Run burn in
        # p0_new, _, _ = sampler.run_mcmc(p0, self.nburn_in)
        p0_new = p0

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
            self.converged = np.all(tau * self.n_indep_samples < (x + 1) * 100)
            self.converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if self.converged:
                print('Done')
                self.index_conv_i = index
                self.autocorr_i = autocorr
                break
            old_tau = tau

        if not self.converged:
            print('Sampler did not converge.')

        nburn_in_steps = 100 #int(self.nburn_in * np.nanmax(tau))

        # Obtain chain of samples and discard burn in.
        self.chain = sampler.chain[:, nburn_in_steps:, :]  # [walkers,steps,variables]
        self.flat_samples = sampler.chain[:, nburn_in_steps:, :].reshape((-1, self.ndim))

    def calculate_and_plot_results_bayesian(self):

        ln_ms_age = self.flat_samples[:, 0]
        ln_cooling_age = self.flat_samples[:, 1]
        delta_m = self.flat_samples[:, 2]

        # Calculate the dependent parameters from the independent parameters
        ln_total_age = np.log10(10 ** ln_ms_age + 10 ** ln_cooling_age)

        f_initial_mass, model_initial_mass, ms_age_model = self.isochrone_model

        min_initial_mass_mist = np.nanmin(model_initial_mass)
        max_initial_mass_mist = np.nanmax(model_initial_mass)

        initial_mass = f_initial_mass(ln_ms_age)
        final_mass = ifmr_bayesian(initial_mass, self.model_ifmr,
                                   min_initial_mass_mist, max_initial_mass_mist)

        final_mass += delta_m

        len1 = len(ln_ms_age[~np.isnan(ln_ms_age)])
        len2 = len(final_mass[~np.isnan(final_mass)])
        assert len1 == len2, f"{len1} not the same as {len2} "

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

        results_i = calc_percentiles(res_ms_age, res_cool_age, res_tot_age,
                                     initial_mass, final_mass,
                                     self.high_perc, self.low_perc)

        if self.display_plots or self.save_plots:
            # Plot corner plot with results from EMCEE for MS age, Total age,
            # and delta m.
            self.plot_results_mcmc_traces()
            self.plot_results_mcmc_corner()

            # Plot distribution for all the white dwarf and progenitor
            # parameters
            if self.datatype == 'yr':
                r_dummy = calc_percentiles(ln_ms_age, ln_cooling_age,
                                           ln_total_age, initial_mass,
                                           final_mass, self.high_perc,
                                           self.low_perc)
                plot_distributions(ln_ms_age, ln_cooling_age, ln_total_age,
                                   initial_mass, final_mass, self.datatype,
                                   r_dummy, self.display_plots, self.save_plots,
                                   name=self.wd_path_id)
            else:
                plot_distributions(res_ms_age, res_cool_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   results_i, self.display_plots,
                                   self.save_plots, name=self.wd_path_id)

            # Plot autocorrelation as a function of steps to confirm convergence
            if self.converged:
                self.plot_autocorr()

        if self.save_plots:
            file = open(self.wd_path_id + "_distributions.txt", 'a')
            file.write('#ms_age\tcool_age\ttotal_age\tinitial_mass\tfinal_mass\n')
            for x1, x2, x3, x4, x5 in zip(res_ms_age, res_cool_age, res_tot_age,
                                          initial_mass, final_mass):
                file.write(str(x1) + '\t' + str(x2) + '\t' + str(x3) + '\t' +
                           str(x4) + '\t' + str(x5) + '\n')
            file.close()

        return results_i

    def plot_autocorr(self):
        N = 100 * np.arange(1, self.index_conv_i + 1)
        plt.plot(N, N / 100.0, "--k", label=r"$\tau = N/100$")
        plt.loglog(N, self.autocorr_i[:self.index_conv_i], "-")
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"mean $\hat{\tau}$")
        plt.legend(fontsize=14)
        plt.grid()
        if self.save_plots:
            plt.savefig(self.wd_path_id + '_corr_time.png')
        if self.display_plots:
            plt.show()
        plt.close()

    def plot_results_mcmc_traces(self):

        labels = [r'$\log_{10}(t_{\rm ms}/{\rm yr})$',
                  r'$\log_{10}(t_{\rm cool}/{\rm yr})$', r'$\Delta_{\rm m}$']
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        for ax, j, label in zip([ax1, ax2, ax3], range(3), labels):
            for i in range(50):
                ax.plot(self.chain[i, :, j], color='k', alpha=0.05)
                ax.axhline(y=np.median(self.flat_samples[:, j]),
                           color='tab:blue')
            ax.set_ylabel(label)
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.wd_path_id + '_trace.png')
        if self.display_plots:
            plt.show()
        plt.close(f)

    def plot_results_mcmc_corner(self):
        labels = [r'$\log_{10}(t_{\rm ms}/{\rm yr})$',
                  r'$\log_{10}(t_{\rm cool}/{\rm yr})$', r'$\Delta _{\rm m}$']

        fig = corner.corner(self.flat_samples, labels=labels,
                            quantiles=[.16, .50, .84],
                            show_titles=True, title_kwargs={"fontsize": 12})
        if self.save_plots:
            fig.savefig(self.wd_path_id + '_corner_plot.png', dpi=300)
        if self.display_plots:
            plt.show()
        plt.close(fig)

    def check_teff_logg(self):

        for x, y, z, w in zip(self.teff, self.e_teff, self.logg,
                              self.e_logg):
            self.teff_i = x
            self.e_teff_i = y
            self.logg_i = z
            self.e_logg_i = w
            approx = self.calc_single_star_params()
            cool_age, final_mass, initial_mass, ms_age = approx

            if np.isnan(final_mass + cool_age) or cool_age < 5.477:
                text = "Effective temperature and/or surface " \
                       "gravity are outside of the allowed values of " \
                       "the model. Teff: %0.2f, logg: %1.2f" % (x, z)
                warnings.warn(text)
            elif ~np.isnan(final_mass) and np.isnan(initial_mass):
                text = "Final mass " \
                       "is outside the range allowed " \
                       "by the IFMR. Cannot estimate initial mass, main " \
                       "sequence age or total age. " \
                       "Teff: %0.2f, logg: %1.2f, Final mass ~ %2.2f Msun " % (x, z, final_mass)
                warnings.warn(text)
            elif (~np.isnan(final_mass + cool_age + initial_mass)
                  and np.isnan(ms_age)):
                text = "Initial mass " \
                       "is outside of the range " \
                       "allowed by the MIST isochrones. Cannot estimate " \
                       "main sequence age or total age. " \
                       "Teff: %0.2f, logg: %1.2f, Initial mass ~ %2.2f Msun " % (x, z, initial_mass)
                warnings.warn(text)
            elif final_mass < 0.56 or final_mass > 1.2414:
                text = "The IFMR is going to be extrapolated to " \
                       "calculate initial mass, main sequence " \
                       "age and total age. Use these parameters with " \
                       "caution. " \
                       "Teff: %0.2f, logg: %1.2f, Final mass ~ %2.2f Msun " % (x, z, final_mass)
                warnings.warn(text)

    def calc_wd_age_fast_test(self):
        """
        Calculated white dwarfs ages with a fast_test approach. Starts from normal
        distribution of teff and logg based on the errors and passes the full
        distribution through the same process to get a distribution of ages.
        """

        # Set up the distribution of teff and logg
        teff_dist, logg_dist = [], []
        for teff_i, e_teff_i, logg_i, e_logg_i in zip(self.teff, self.e_teff,
                                                      self.logg, self.e_logg):
            if np.isnan(teff_i + e_teff_i + logg_i + e_logg_i):
                teff_dist.append(np.nan)
                logg_dist.append(np.nan)
            else:
                teff_dist.append(np.random.normal(teff_i, e_teff_i, self.n_mc))
                logg_dist.append(np.random.normal(logg_i, e_logg_i, self.n_mc))
        teff_dist, logg_dist = np.array(teff_dist), np.array(logg_dist)

        # From teff and logg estimate cooling age and final mass
        res = calc_cooling_age(teff_dist, logg_dist, model=self.model_wd)
        log_cooling_age_dist, final_mass_dist = res

        # From final mass estimate initial mass
        initial_mass_dist = calc_initial_mass(self.model_ifmr, final_mass_dist)

        # From initial mass estimate main sequence age
        log_ms_age_dist = calc_ms_age(initial_mass_dist, feh=self.feh,
                                      vvcrit=self.vvcrit)

        # Estimate total age adding cooling age and main sequence age
        log_total_age_dist = np.log10(10 ** log_cooling_age_dist
                                      + 10 ** log_ms_age_dist)

        # Calculate percentiles and save results
        if self.datatype == 'Gyr':
            ms_age_dummy = 10 ** log_ms_age_dist / 1e9
            cooling_age_dummy = 10 ** log_cooling_age_dist / 1e9
            total_age_dummy = 10 ** log_total_age_dist / 1e9
        elif self.datatype == 'yr':
            ms_age_dummy = 10 ** log_ms_age_dist
            cooling_age_dummy = 10 ** log_cooling_age_dist
            total_age_dummy = 10 ** log_total_age_dist
        else:
            ms_age_dummy = log_ms_age_dist
            cooling_age_dummy = log_cooling_age_dist
            total_age_dummy = log_total_age_dist

        self.results_fast_test = Table()
        labels = ['ms_age', 'cooling_age', 'total_age', 'initial_mass',
                  'final_mass']
        self.distributions = [ms_age_dummy, cooling_age_dummy, total_age_dummy,
                              initial_mass_dist, final_mass_dist]

        for label, dist in zip(labels, self.distributions):
            median, high_err, low_err = calc_dist_percentiles(dist,
                                                              self.high_perc,
                                                              self.low_perc)
            self.results_fast_test[label + '_median'] = median
            self.results_fast_test[label + '_err_low'] = low_err
            self.results_fast_test[label + '_err_high'] = high_err

        if self.return_distributions:
            for label, dist in zip(labels, self.distributions):
                self.results_fast_test[label + '_dist'] = dist

        if self.save_plots or self.display_plots:
            self.plot_distributions_fast_test()

    def plot_distributions_fast_test(self):
        # Plot resulting distributions

        for i in range(len(self.teff)):

            x1 = self.teff[i]
            x2 = self.logg[i]
            x3 = np.array(self.distributions[0][i])
            x4 = np.array(self.distributions[1][i])
            x5 = np.array(self.distributions[2][i])
            x6 = np.array(self.distributions[3][i])
            x7 = np.array(self.distributions[4][i])
            x8 = self.results_fast_test[i]

            # Set name of path and wd models to identif results
            self.teff_i = x1
            self.logg_i = x2
            self.wd_path_id = self.get_wd_path_id()

            print(f'Running teff:{x1} logg:{x2}')

            if self.datatype == 'yr':
                x3, x4, x5 = np.log10(x3), np.log10(x4), np.log10(x5)
                res_dummy = calc_percentiles(x3, x4, x5, x6, x7,
                                             self.high_perc, self.low_perc)
                plot_distributions(x3, x4, x5, x6, x7, self.datatype, res_dummy,
                                   self.display_plots, self.save_plots,
                                   name=self.wd_path_id + '_fast_test')
            else:
                plot_distributions(x3, x4, x5, x6, x7, self.datatype, x8,
                                   self.display_plots, self.save_plots,
                                   name=self.wd_path_id + '_fast_test')
