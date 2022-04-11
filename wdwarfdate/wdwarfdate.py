import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import os
from .cooling_age import get_cooling_model_grid
from .ms_age import get_isochrone_model_grid
from .extra_func import calc_percentiles, plot_distributions
from .extra_func import calc_dist_percentiles, calc_single_star_params
from .fast_test_age import estimate_parameters_fast_test
from .bayesian_age_grid import get_other_params, get_dist_parameters, \
    get_idx_sample, calc_posterior_grid


class WhiteDwarf:
    def __init__(self, teff0, e_teff0, logg0, e_logg0, method='bayesian',
                 model_wd='DA', feh='p0.00', vvcrit='0.0',
                 model_ifmr='Cummings_2018_MIST', high_perc=84, low_perc=16,
                 datatype='yr', path='results/', n_mc=2000,
                 n_mi=256, n_log10_tcool=256, n_delta=128, min_mi='',
                 max_mi='', min_log10_tcool='', max_log10_tcool='',
                 tail=0.005, adjust_tail=True,
                 return_distributions=False, save_plots=False,
                 display_plots=True):
        """
        Parameters
        ----------
        teff0 : scalar, array. Effective temperature of the white dwarf
        e_teff0 : scalar, array. Error in the effective temperature of the white
        dwarf
        logg0 : scalar, array. Surface gravity of the white dwarf
        e_logg0 : scalar, array. Error in surface gravity of the white dwarf
        method : string. 'bayesian' or 'fast_test'. Bayesian will the grid
                 method and output the results. fast_test runs a normal
                 distribution centered at the value with a std of the error
                 through all the models chosen.
        model_wd : string. Spectral type of the white dwarf 'DA' or 'non-DA'.
        feh : string. Parameter for the isochrone. Can be: 'm4.00','m1.00',
              'p0.00' or 'p0.50'
        vvcrit : string. Parameter for the isochrone. Can be: '0.0' or '0.4'
        model_ifmr : string. Initial to final mass relation model. Can be
                     'Cummings_2018_MIST', 'Cummings_2018_PARSEC',
                     'Salaris_2009' or 'Williams_2009'.
        high_perc : scalar. Percentage at which the high errors will be
                    calculated.
        low_perc : scalar. Percentage at which the low errors will be
                   calculated.
        datatype : string. 'yr', 'Gyr' or 'log'. Units in which the results
                   will be output.
        path : string. Name of the folder where all the plots and distribution
               file will be save. If it doesn't exist, the code will create it.
        n_mc : scalar. Length of the distribution for each parameter. Only
               useful in fast_test mode.
        n_mi : scalar. Number of bins for the initial mass axis of the grid.
        n_log10_tcool : scalar. Number of bins for the cooling age axis of
                        the grid.
        n_delta: scalar. Number of bins for the delta parameter axis of the
                 grid.
        min_mi : scalar. Minimum limit for the the initial mass axis of the
                 grid.
        max_mi : scalar. Maximum limit for the the initial mass axis of the
                 grid.
        min_log10_tcool : scalar. Minimum limit for the the cooling age axis
                          of the grid.
        max_log10_tcool : scalar. Maximum limit for the the cooling age axis
                          of the grid.
        tail : scalar. Percentage cut off for log probability: 0.95 would
               cut the probability close to the highest probability value,
               0 does not remove anything.
        adjust_tail : True or False. If True, the limits of the grid will be
                      adjusted automatically.
        return_distributions : True or False. In the fast test method adds
                               columns to the table with results with
                               the distributions of each parameter. In the
                               Bayesian method it creates a new attribute for
                               the object WhiteDwarfs with the distributions
                               for the total age only.
        save_plots : True or False. If True, plots and saves the figures
                   describing the result in the path given.
        display_plots : True or False. If True, will display plots after
                        making them. Good option if working in jupyter
                        notebook.
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
        self.n_mc = n_mc
        self.max_age = np.log10(15 * 1e9)
        self.return_distributions = return_distributions
        # Bayesian method objects.
        if self.method == 'bayesian':
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
            for x, y, z, w in zip(self.teff, self.e_teff, self.logg,
                                  self.e_logg):
                print(f'Running Teff = {np.round(x, 2)}'
                      + r' +/- '
                      + f'{np.round(y, 2)} K, '
                      + f'logg = {np.round(z, 2)}'
                      + r' +/- '
                      + f'{np.round(w, 2)}')
                self.teff_i = x
                self.e_teff_i = y
                self.logg_i = z
                self.e_logg_i = w

                if np.isnan(x + y + z + w):
                    print('Warning: teff, logg or their uncertainties are ' +
                          'nans.')
                    results_i = np.ones(15) * np.nan
                else:
                    # Set name of path and wd models to identify results
                    self.wd_path_id = self.get_wd_path_id()

                    results_i = self.calc_wd_age_bayesian()

                self.results.add_row(results_i)

        elif self.method == 'fast_test':
            self.calc_wd_age_fast_test()

    def calc_wd_age_bayesian(self):

        r = calc_single_star_params(self.teff_i, self.logg_i, self.model_wd,
                                    self.model_ifmr, self.feh, self.vvcrit)
        cool_age, final_mass, initial_mass, ms_age = r

        if final_mass > 1.1 or final_mass < 0.45:
            print('Warning: Final mass is outside the range normally ' +
                  'considered as single star evolution (0.45-1.1 Msun).')

        if initial_mass > 5 and self.model_ifmr == 'Cummings_2018_MIST':
            print('Warning: Consider using the PARSEC-based IFMR for '
                  'progenitor stars more massive than 5 MSun as recommended '
                  'by Cummings et al. (2018) because the MIST models tend to '
                  'underestimate the mass of the progenitor star.')

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
            if(final_mass < 0.56
                    or (final_mass > 1.2497
                        and self.model_ifmr == 'Cummings_2018_PARSEC')):
                print("Warning: The IFMR is going to be extrapolated to " +
                      "calculate initial mass, main sequence " +
                      "age and total age. Use these parameters with " +
                      f"caution. Final mass ~ {np.round(final_mass, 2)} Msun ")

            results_i = self.calc_bayesian_wd_age_grid()

        return results_i

    def get_wd_path_id(self):
        # Set the name to identify the results from each white dwarf
        tg_name = 'teff_' + str(self.teff_i) + '_logg_' \
                  + str(np.round(self.logg_i,2))
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

        self.params, params_grid, self.posterior = res
        self.params_prob = [np.nansum(self.posterior, axis=(1, 2)),
                            np.nansum(self.posterior, axis=(0, 2)),
                            np.nansum(self.posterior, axis=(0, 1))]

        sample_idx = get_idx_sample(self.posterior)
        r = get_dist_parameters(params_grid[0], sample_idx)
        self.mi_sample = r[0]
        self.mi_median = r[1]
        self.mi_err_high = r[2]
        self.mi_err_low = r[3]
        r = get_dist_parameters(params_grid[1], sample_idx)
        self.log10_tcool_sample = r[0]
        self.log10_tcool_median = r[1]
        self.log10_tcool_err_high = r[2]
        self.log10_tcool_err_low = r[3]
        r = get_dist_parameters(params_grid[2], sample_idx)
        self.delta_m_sample = r[0]
        self.delta_m_median = r[1]
        self.delta_m_err_high = r[2]
        self.delta_m_err_low = r[3]

        r = get_other_params(self.mi_sample, self.log10_tcool_sample,
                             self.delta_m_sample, self.models0)
        self.mf_sample = r[0]
        self.log10_tms_sample = r[1]
        self.log10_ttot_sample = r[2]

        results_plot = calc_percentiles(self.log10_tms_sample,
                                        self.log10_tcool_sample,
                                        self.log10_ttot_sample, self.mi_sample,
                                        self.mf_sample,
                                        self.high_perc, self.low_perc)

        if self.datatype == 'yr':
            tms_sample_dummy = self.log10_tms_sample
            tcool_sample_dummy = self.log10_tcool_sample
            ttot_sample_dummy = self.log10_ttot_sample
            results_i = calc_percentiles(10 ** self.log10_tms_sample,
                                         10 ** self.log10_tcool_sample,
                                         10 ** self.log10_ttot_sample,
                                         self.mi_sample,
                                         self.mf_sample,
                                         self.high_perc, self.low_perc)
        elif self.datatype == 'Gyr':
            tms_sample_dummy = (10 ** self.log10_tms_sample) / 1e9
            tcool_sample_dummy = (10 ** self.log10_tcool_sample) / 1e9
            ttot_sample_dummy = (10 ** self.log10_ttot_sample) / 1e9
            results_i = calc_percentiles(tms_sample_dummy,
                                         tcool_sample_dummy,
                                         ttot_sample_dummy, self.mi_sample,
                                         self.mf_sample,
                                         self.high_perc, self.low_perc)
            results_plot = results_i
        else:
            tms_sample_dummy = self.log10_tms_sample
            tcool_sample_dummy = self.log10_tcool_sample
            ttot_sample_dummy = self.log10_ttot_sample
            results_i = results_plot

        if self.save_plots or self.display_plots:
            self.make_grid_plot(self.mi_median, self.log10_tcool_median,
                                self.delta_m_median, self.mi_err_low,
                                self.log10_tcool_err_low,
                                self.delta_m_err_low,
                                self.mi_err_high, self.log10_tcool_err_high,
                                self.delta_m_err_high,
                                self.params, self.params_prob, self.posterior)

            plot_distributions(tms_sample_dummy, tcool_sample_dummy,
                               ttot_sample_dummy, self.mi_sample,
                               self.mf_sample,
                               datatype=self.datatype, results=results_plot,
                               display_plots=self.display_plots,
                               save_plots=self.save_plots,
                               path = self.wd_path_id)

        if self.return_distributions:
            self.distributions.append([tms_sample_dummy, tcool_sample_dummy,
                                       ttot_sample_dummy, self.mi_sample,
                                       self.mf_sample,
                                       self.delta_m_sample])

        return results_i

    def make_grid_plot(self, mi_median, log10_tcool_median, delta_m_median,
                       mi_err_low, log10_tcool_err_low, delta_m_err_low,
                       mi_err_high, log10_tcool_err_high, delta_m_err_high,
                       params, params_prob, posterior):

        params_label = [r'$m_{\rm ini}$', r'$\log _{10} t_{\rm cool}$',
                        r'$\Delta _{\rm m}$']
        res = [mi_median, log10_tcool_median, delta_m_median]
        res_err_low = [mi_err_low, log10_tcool_err_low, delta_m_err_low]
        res_err_high = [mi_err_high, log10_tcool_err_high, delta_m_err_high]
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
            plt.savefig(self.wd_path_id + '_gridplot.png', dpi=300)
        if self.display_plots:
            plt.show()
        plt.close(f)

    def calc_final_mass_cooling_age(self):
        """
        This function estimates final mass and cooling age in the Bayesian
        method, in the case that the initial mass, main sequence age and
        total age cannot be estimated because the final mass is outside
        the calibrated regime of the IFMR.

        """

        distributions = estimate_parameters_fast_test(np.array([self.teff_i]),
                                                      np.array([self.e_teff_i]),
                                                      np.array([self.logg_i]),
                                                      np.array([self.e_logg_i]),
                                                      self.model_ifmr,
                                                      self.model_wd, self.feh,
                                                      self.vvcrit, self.n_mc,
                                                      self.max_age,
                                                      warning=0)

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
            self.distributions.append([res_ms_age, log_cooling_age, res_tot_age,
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
                                   path=self.wd_path_id)
            else:
                plot_distributions(res_ms_age, res_cool_age, res_tot_age,
                                   initial_mass, final_mass, self.datatype,
                                   results_i, self.display_plots,
                                   self.save_plots, path=self.wd_path_id)

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
                                   path=self.wd_path_id + '_fast_test')
            else:
                plot_distributions(x3, x4, x5, x6, x7, self.datatype, x8,
                                   self.display_plots, self.save_plots,
                                   path=self.wd_path_id + '_fast_test')
