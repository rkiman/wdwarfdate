import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import os
import emcee
import corner
import warnings
import time
from .cooling_age import calc_cooling_age, get_cooling_model, \
    get_cooling_model_grid
from .ifmr import calc_initial_mass, ifmr_bayesian
from .ms_age import calc_ms_age, get_isochrone_model, get_isochrone_model_grid
from .extra_func import calc_percentiles, plot_distributions
from .extra_func import calc_dist_percentiles, calc_single_star_params
from .bayesian_age import ln_posterior_prob
from .check_convergence import calc_auto_corr_time
from .fast_test_age import estimate_parameters_fast_test, check_teff_logg
from .bayesian_age_grid import get_other_params, get_dist_parameters, \
    get_idx_sample, calc_posterior_grid


class WhiteDwarf:
    def __init__(self, teff0, e_teff0, logg0, e_logg0, method='bayesian_grid',
                 model_wd='DA', feh='p0.00', vvcrit='0.0',
                 model_ifmr='Cummings_2018_MIST', init_params=[],
                 high_perc=84,
                 low_perc=16, datatype='yr', path='results/', nburn_in=2,
                 max_n=100000, n_indep_samples=100, n_mc=2000,
                 n_mi=256, n_log10_tcool=256, n_delta=128, min_mi='', max_mi='',
                 min_log10_tcool='', max_log10_tcool='',
                 tail=0.005, adjust_tail=True,
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
        self.max_age = np.log10(15 * 1e9)
        if not init_params:
            if self.n > 1:
                self.init_params = [[] for i in range(self.n)]
            else:
                self.init_params = [[]]
        else:
            self.init_params = init_params
        self.init_params_i = []
        # Bayesian method objects.
        if self.method == 'bayesian_emcee':
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
        elif self.method == 'bayesian_grid':
            self.return_distributions = return_distributions
            self.distributions = []
            self.set_values = [min_mi, max_mi,
                               min_log10_tcool, max_log10_tcool,
                               n_mi, n_log10_tcool, n_delta,
                               tail, adjust_tail]
            if all([x != '' for x in self.set_values]):
                self.adjust_tail = False

            self.results = Table(
                names=('ms_age_median', 'ms_age_err_low', 'ms_age_err_high',
                       'cooling_age_median', 'cooling_age_err_low',
                       'cooling_age_err_high', 'total_age_median',
                       'total_age_err_low', 'total_age_err_high',
                       'initial_mass_median', 'initial_mass_err_low',
                       'initial_mass_err_high', 'final_mass_median',
                       'final_mass_err_low', 'final_mass_err_high'))

        # Fast-test method objects.
        elif self.method == 'fast_test':
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

        if self.method == 'bayesian_grid' or self.method == 'bayesian_emcee':
            if self.save_log:
                if os.path.exists(self.path + 'run_log.txt'):
                    file_log = open(self.path + 'run_log.txt', 'a')
                else:
                    file_log = open(self.path + 'run_log.txt', 'a')
                    file_log.write('Teff\tlogg\ttime (min)\tconverged\twarning\n')
            for x, y, z, w, q in zip(self.teff, self.e_teff, self.logg,
                                     self.e_logg, self.init_params):
                print(f'Running Teff = {np.round(x, 2)} K, ' +
                      f'logg = {np.round(z, 2)}')
                start = time.time()
                self.teff_i = x
                self.e_teff_i = y
                self.logg_i = z
                self.e_logg_i = w
                self.init_params_i = q
                self.converged = False
                self.warn_text = ''

                # Set name of path and wd models to identify results
                self.wd_path_id = self.get_wd_path_id()

                results_i = self.calc_wd_age_bayesian()

                self.results.add_row(results_i)
                end = time.time()
                if self.save_log:
                    file_log.write(str(x) + '\t' + str(z) + '\t' +
                                   str((end - start) / 60) + '\t' +
                                   str('Y' if self.converged else 'N') + '\t' +
                                   self.warn_text + '\n')

        elif self.method == 'fast_test':
            self.calc_wd_age_fast_test()

    def calc_wd_age_bayesian(self):

        r = calc_single_star_params(self.teff_i, self.logg_i, self.model_wd,
                                    self.model_ifmr, self.feh, self.vvcrit)
        cool_age, final_mass, initial_mass, ms_age = r

        if np.isnan(cool_age + final_mass) or cool_age < np.log10(3e5):
            print("Warning: Effective temperature and/or surface " +
                  "gravity are outside of the allowed values of " +
                  "the models.")
            results_i = np.ones(15) * np.nan
        elif ~np.isnan(final_mass) and np.isnan(initial_mass):
            print("Warning: Final mass is outside the range allowed " +
                  "by the IFMR. Cannot estimate initial mass, main " +
                  "sequence age or total age. Final mass and " +
                  "cooling age were calculated with the fast_test " +
                  f"method. Final mass ~ {np.round(final_mass, 2)} Msun ")
            # Calculate final mass and cooling age with fast_test method
            results_i = self.calc_final_mass_cooling_age()
        elif (~np.isnan(final_mass + cool_age + initial_mass)
              and np.isnan(ms_age)):
            print("Warning: Initial mass is outside of the range " +
                  "allowed by the MIST isochrones. Cannot estimate " +
                  "main sequence age or total age. Run the fast_test " +
                  "method to obtain a result for the rest of the " +
                  "parameters. " +
                  f"Initial mass ~ {np.round(initial_mass, 2)} Msun ")
            results_i = self.calc_final_mass_cooling_age()
        else:
            if final_mass < 0.56:
                print("Warning: The IFMR is going to be extrapolated to " +
                      "calculate initial mass, main sequence " +
                      "age and total age. Use these parameters with " +
                      f"caution. Final mass ~ {np.round(final_mass, 2)} Msun ")

            if self.method == 'bayesian_emcee':
                results_i = self.calc_bayesian_wd_age_emcee()
            else:
                results_i = self.calc_bayesian_wd_age_grid()

        return results_i

    def get_wd_path_id(self):
        # Set the name to identify the results from each white dwarf
        tg_name = 'teff_' + str(self.teff_i) + '_logg_' + str(self.logg_i)
        mist_name = '_feh_' + self.feh + '_vvcrit_' + self.vvcrit
        models_name = mist_name + '_' + self.model_wd + '_' + self.model_ifmr
        return self.path + tg_name + models_name

    def calc_bayesian_wd_age_grid(self):
        log10_tcool_min_wide_range = np.log10(0.3 * 1e6)
        log10_tcool_max_wide_range = self.max_age
        mi_min_wide_range = 0.1
        mi_max_wide_range = 10

        self.n_mi = self.set_values[4]
        self.n_log10_tcool = self.set_values[5]
        self.n_delta = self.set_values[6]
        self.min_mi = self.set_values[0]
        self.max_mi = self.set_values[1]
        self.min_log10_tcool = self.set_values[2]
        self.max_log10_tcool = self.set_values[3]
        self.tail = self.set_values[7]
        self.adjust_tail = self.set_values[8]

        self.cooling_models = get_cooling_model_grid(self.model_wd)
        self.isochrone_model = get_isochrone_model_grid(feh=self.feh,
                                                        vvcrit=self.vvcrit)

        self.models0 = [self.model_ifmr, self.isochrone_model,
                        self.cooling_models]

        # If adjust_tail = True, it looks for the narrow range where the
        # posterior has higher probability. The code will adjust the limits
        # which are not set by the user. If adjust_tail = False it will use a
        # large range for the grid to evaluate the posterior, or the value
        # set by the user.
        if self.adjust_tail:
            # If any of the limits was set by the user, it uses that limit.
            # If not it performs a wide range evaluation of the posterior.
            # Limits for low and high temperatures were set according to the
            # cooling models to avoid grid evaluation in regions far from
            # the answer.
            test_lim = []
            if self.teff_i < 4000:
                print('Warning: You might need to adjust min_mi, max_mi, '
                      'min_log10_tcool and max_log10_tcool manually to obtain '
                      'a better evaluation of the posterior.')
                default_values = [mi_min_wide_range, mi_max_wide_range,
                                  8, log10_tcool_max_wide_range]
            elif self.teff_i > 1.5e4:
                default_values = [mi_min_wide_range, mi_max_wide_range,
                                  log10_tcool_min_wide_range, 9.6989700043360]
            else:
                default_values = [mi_min_wide_range, mi_max_wide_range,
                                  log10_tcool_min_wide_range,
                                  log10_tcool_max_wide_range]

            for value, default in zip(self.set_values[:4], default_values):
                if value == '':
                    test_lim.append(default)
                else:
                    test_lim.append(value)

            assert all([x != '' for x in test_lim])

            res = calc_posterior_grid(self.teff_i, self.e_teff_i, self.logg_i,
                                      self.e_logg_i, self.models0,
                                      n_mi=256, n_log10_tcool=256,
                                      n_delta=self.n_delta, min_mi=test_lim[0],
                                      max_mi=test_lim[1],
                                      min_log10_tcool=test_lim[2],
                                      max_log10_tcool=test_lim[3],
                                      max_log10_age = log10_tcool_max_wide_range)
            params_test, params_grid_test, posterior_test = res

            params_prob_test = [np.nansum(posterior_test, axis=(1, 2)),
                                np.nansum(posterior_test, axis=(0, 2)),
                                np.nansum(posterior_test, axis=(0, 1))]

            # Using the wide search for the posterior, we define the limits
            # of the grid according to the chosen (or set) tail tolerance.
            min_lim_prob_mi = self.tail * np.nanmax(params_prob_test[0])
            min_lim_prob_tcool = self.tail * np.nanmax(params_prob_test[1])
            prob_bigger_zero_mi = params_prob_test[0] > min_lim_prob_mi
            prob_bigger_zero_tcool = params_prob_test[1] > min_lim_prob_tcool
            # initial mass
            mi_array_clean = params_test[0][prob_bigger_zero_mi]
            if self.min_mi == '':
                self.min_mi = np.min(mi_array_clean)
            if self.max_mi == '':
                self.max_mi = np.max(mi_array_clean)
            # total age
            tcool_array_clean = params_test[1][prob_bigger_zero_tcool]
            if self.min_log10_tcool == '':
                self.min_log10_tcool = np.min(tcool_array_clean)
            if self.max_log10_tcool == '':
                self.max_log10_tcool = np.max(tcool_array_clean)
        # If adjust_tail is false, use the the wide range for the evaluation
        # of the posterior, or the values set by the user, if any.
        else:
            if self.min_mi == '':
                self.min_mi = mi_min_wide_range
            if self.max_mi == '':
                self.max_mi = mi_max_wide_range
            if self.min_log10_tcool == '':
                self.min_log10_tcool = log10_tcool_min_wide_range
            if self.max_log10_tcool == '':
                self.max_log10_tcool = log10_tcool_max_wide_range

        print(f'Grid limits used to evaluate the posterior: ' +
              f'mi = {np.round(self.min_mi, 2)}-{np.round(self.max_mi, 2)} ' +
              f'Msun, log10_tcool = {np.round(self.min_log10_tcool, 2)}-' +
              f'{np.round(self.max_log10_tcool, 2)}')
        assert all([self.min_mi != '', self.max_mi != '',
                    self.min_log10_tcool != '', self.max_log10_tcool != ''])

        # Calculate the posterior using a grid with the set limits
        res = calc_posterior_grid(self.teff_i, self.e_teff_i, self.logg_i,
                                  self.e_logg_i, self.models0,
                                  n_mi=self.n_mi,
                                  n_log10_tcool=self.n_log10_tcool,
                                  n_delta=self.n_delta, min_mi=self.min_mi,
                                  max_mi=self.max_mi,
                                  min_log10_tcool=self.min_log10_tcool,
                                  max_log10_tcool=self.max_log10_tcool,
                                  max_log10_age = log10_tcool_max_wide_range)

        params, params_grid, posterior = res
        params_prob = [np.nansum(posterior, axis=(1, 2)),
                       np.nansum(posterior, axis=(0, 2)),
                       np.nansum(posterior, axis=(0, 1))]

        sample_idx = get_idx_sample(posterior)
        r = get_dist_parameters(params_grid[0], sample_idx)
        mi_sample, mi_median, mi_err_high, mi_err_low = r
        r = get_dist_parameters(params_grid[1], sample_idx)
        log10_tcool_sample, log10_tcool_median, log10_tcool_err_high, log10_tcool_err_low = r
        r = get_dist_parameters(params_grid[2], sample_idx)
        delta_m_sample, delta_m_median, delta_m_err_high, delta_m_err_low = r
        r = get_other_params(mi_sample, log10_tcool_sample, delta_m_sample,
                             self.models0)
        mf_sample, log10_tms_sample, log10_ttot_sample = r

        results_plot = calc_percentiles(log10_tms_sample,
                                        log10_tcool_sample,
                                        log10_ttot_sample, mi_sample,
                                        mf_sample,
                                        self.high_perc, self.low_perc)

        if self.datatype == 'yr':
            tms_sample_dummy = log10_tms_sample
            tcool_sample_dummy = log10_tcool_sample
            ttot_sample_dummy = log10_ttot_sample
            results_i = calc_percentiles(10 ** log10_tms_sample,
                                         10 ** log10_tcool_sample,
                                         10 ** log10_ttot_sample, mi_sample,
                                         mf_sample,
                                         self.high_perc, self.low_perc)
        elif self.datatype == 'Gyr':
            tms_sample_dummy = (10 ** log10_tms_sample) / 1e9
            tcool_sample_dummy = (10 ** log10_tcool_sample) / 1e9
            ttot_sample_dummy = (10 ** log10_ttot_sample) / 1e9
            results_i = calc_percentiles(tms_sample_dummy,
                                         tcool_sample_dummy,
                                         ttot_sample_dummy, mi_sample,
                                         mf_sample,
                                         self.high_perc, self.low_perc)
            results_plot = results_i
        else:
            tms_sample_dummy = log10_tms_sample
            tcool_sample_dummy = log10_tcool_sample
            ttot_sample_dummy = log10_ttot_sample
            results_i = results_plot

        if self.save_plots or self.display_plots:
            self.make_grid_plot(mi_median, log10_tcool_median, delta_m_median,
                                mi_err_low, log10_tcool_err_low,
                                delta_m_err_low,
                                mi_err_high, log10_tcool_err_high,
                                delta_m_err_high,
                                params, params_prob, posterior)

            plot_distributions(tms_sample_dummy, tcool_sample_dummy,
                               ttot_sample_dummy, mi_sample, mf_sample,
                               datatype=self.datatype, results=results_plot,
                               display_plots=True, save_plots=False)

        if self.return_distributions:
            self.distributions.append([tms_sample_dummy, tcool_sample_dummy,
                                       ttot_sample_dummy, mi_sample, mf_sample,
                                       delta_m_sample])

        return results_i

    def make_grid_plot(self, mi_median, log10_tott_median, delta_m_median,
                       mi_err_low, log10_tott_err_low, delta_m_err_low,
                       mi_err_high, log10_tott_err_high, delta_m_err_high,
                       params, params_prob, posterior):

        params_label = [r'$m_{\rm ini}$', r'$\log _{10} t_{\rm cool}$',
                        r'$\Delta m$']
        res = [mi_median, log10_tott_median, delta_m_median]
        res_err_low = [mi_err_low, log10_tott_err_low, delta_m_err_low]
        res_err_high = [mi_err_high, log10_tott_err_high, delta_m_err_high]
        title = r"${{{0:.2f}}}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
        f, axs = plt.subplots(3, 3, figsize=(10, 10), sharex='col')

        for i in range(3):
            for j in range(3):
                if i == j:
                    axs[i, j].plot(params[i], params_prob[i], '.-')
                    axs[i, j].axvline(x=res[i], color='k')
                    axs[i, j].axvline(x=res[i] - res_err_low[i], color='k', linestyle='--')
                    axs[i, j].axvline(x=res[i] + res_err_high[i], color='k', linestyle='--')
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_ylim(0,)
                    if any(np.array([np.round(res[i], 2), np.round(res_err_low[i], 2),
                                     np.round(res_err_high[i], 2)]) == 0):
                        dec_num = 2
                        while any(np.array([np.round(res[i], dec_num),
                                            np.round(res_err_low[i], dec_num),
                                            np.round(res_err_high[i], dec_num)]) == 0):
                            dec_num += 1
                            title2 = r"${{{0:." + str(dec_num) + "f}}}_{{-{1:." + str(dec_num) + "f}}}^{{+{2:." + str(
                                dec_num) + "f}}}$"
                            axs[i, j].set_title(params_label[i] + ' = ' + title2.format(np.round(res[i], dec_num),
                                                                                        np.round(res_err_low[i],
                                                                                                 dec_num),
                                                                                        np.round(res_err_high[i],
                                                                                                 dec_num)))
                    else:
                        axs[i, j].set_title(params_label[i] + ' = ' + title.format(np.round(res[i], 2),
                                                                                   np.round(res_err_low[i], 2),
                                                                                   np.round(res_err_high[i], 2)))
                elif i > j:
                    options = np.array([0, 1, 2])
                    mask = np.array([x not in [i, j] for x in options])
                    axis_sum = options[mask][0]
                    axs[i, j].contourf(params[j], params[i], np.nansum(posterior, axis=(axis_sum)).transpose(),
                                       cmap='gist_yarg')
                    if j == 1:
                        axs[i, j].set_yticklabels([])
                else:
                    f.delaxes(axs[i, j])

                if i == 2:
                    axs[i, j].set_xlabel(params_label[j])

                if j == 0 and i != 0:
                    axs[i, j].set_ylabel(params_label[i])

                axs[i, j].tick_params('both', direction='in', top=True, right=True)
                axs[i, j].minorticks_on()
                axs[i, j].tick_params('both', which='minor', direction='in', top=True, right=True)
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.path + '_gridplot.png', dpi=300)
        if self.display_plots:
            plt.show()
        plt.close(f)

    def calc_bayesian_wd_age_emcee(self):
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

        r = calc_single_star_params(self.teff, self.logg, self.model_wd,
                                    self.model_ifmr, self.feh, self.vvcrit)
        cool_age_j, _, _, ms_age_j = r

        init_params = np.array([ms_age_j, cool_age_j, 0])

        if any([np.isnan(x) for x in init_params]):
            teff_dist = [np.random.normal(self.teff_i, self.e_teff_i, self.n_mc)]
            logg_dist = [np.random.normal(self.logg_i, self.e_logg_i, self.n_mc)]
            cool_age, final_mass = calc_cooling_age(teff_dist, logg_dist,
                                                    self.model_wd)
            if self.model_ifmr == 'Marigo_2020':
                ifmr_dummy = 'Cummings_2018_MIST'
            else:
                ifmr_dummy = self.model_ifmr

            initial_mass = calc_initial_mass(ifmr_dummy, final_mass)
            ms_age = calc_ms_age(initial_mass, self.feh, self.vvcrit)

            init_params = np.array([np.nanmedian(ms_age),
                                    np.nanmedian(cool_age), 0])

        return init_params

    def calc_final_mass_cooling_age(self):
        """
        This function estimates final mass and cooling age in the Bayesian
        method, in the case that the initial mass, main sequence age and
        total age cannot be estimated because the final mass is outside
        the calibrated regime of the IFMR.

        """

        distributions = estimate_parameters_fast_test(self.teff_i,
                                                      self.e_teff_i,
                                                      self.logg_i,
                                                      self.e_logg_i,
                                                      self.model_ifmr,
                                                      self.model_wd, self.feh,
                                                      self.vvcrit, self.n_mc,
                                                      self.max_age)

        log_cooling_age = distributions[0][0]
        final_mass = distributions[1][0]
        initial_mass = distributions[2][0]
        res_ms_age = distributions[3][0]
        res_tot_age = distributions[4][0]

        # Adjust units for the ages as required by the user.
        if self.datatype == 'yr':
            res_cool_age = (10 ** log_cooling_age)
        elif self.datatype == 'Gyr':
            res_cool_age = (10 ** log_cooling_age) / 1e9
        else:
            res_cool_age = log_cooling_age

        # Estimate medians and uncertainties for each parameter.
        results_i = calc_percentiles(res_ms_age, res_cool_age, res_tot_age,
                                     initial_mass, final_mass,
                                     self.high_perc, self.low_perc)

        if self.return_distributions:
            self.distributions.append([res_ms_age, res_cool_age, res_tot_age,
                                       initial_mass, final_mass, 0])

        if self.display_plots or self.save_plots:

            # Plot distribution for all the white dwarf and progenitor
            # parameters
            if self.datatype == 'yr':
                r_dummy = calc_percentiles(res_ms_age, log_cooling_age,
                                           res_tot_age, initial_mass,
                                           final_mass, self.high_perc,
                                           self.low_perc)
                plot_distributions(res_ms_age, log_cooling_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   r_dummy, self.display_plots, self.save_plots,
                                   name=self.wd_path_id)
            else:
                plot_distributions(res_ms_age, res_cool_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   results_i, self.display_plots,
                                   self.save_plots, name=self.wd_path_id)

        if self.save_plots:
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

        nburn_in_steps = 100  # int(self.nburn_in * np.nanmax(tau))

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

    def calc_wd_age_fast_test(self):
        """
        Calculated white dwarfs ages with a fast_test approach. Starts from normal
        distribution of teff and logg based on the errors and passes the full
        distribution through the same process to get a distribution of ages.
        """

        distributions = estimate_parameters_fast_test(self.teff, self.e_teff,
                                                      self.logg, self.e_logg,
                                                      self.model_ifmr,
                                                      self.model_wd, self.feh,
                                                      self.vvcrit, self.n_mc,
                                                      self.max_age)

        log_cooling_age_dist = distributions[0]
        final_mass_dist = distributions[1]
        initial_mass_dist = distributions[2]
        log_ms_age_dist = distributions[3]
        log_total_age_dist = distributions[4]

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
