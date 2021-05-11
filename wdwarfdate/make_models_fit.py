import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def fit_cooling_model(model, plot_fit,
                      deg_list=[[6, 6], [6, 6], [8, 7], [7, 7], [7, 8]]):
    if model == 'DA':
        table_model = np.loadtxt(
            '/Users/rociokiman/Documents/wdwarfdate/Models/cooling_models'
            '/Table_DA')
    if model == 'DB':
        table_model = np.loadtxt(
            '/Users/rociokiman/Documents/wdwarfdate/Models/cooling_models'
            '/Table_DB')

    teff_model = table_model[:, 0]
    logg_model = table_model[:, 1]
    age_model = table_model[:, 40]
    mass_model = table_model[:, 2]

    up_lim = 26000

    logg_model_unique = []

    for x in logg_model:
        if x not in logg_model_unique:
            logg_model_unique.append(x)

    logg_model_unique = np.array(logg_model_unique)

    res_age_fit = []
    res_mass_fit = []

    for x, deg in zip(logg_model_unique, deg_list):
        mask_logg = (x == logg_model) * (teff_model < up_lim)

        res_age = np.polyfit(teff_model[mask_logg], age_model[mask_logg],
                             deg=deg[0])
        res_mass = np.polyfit(teff_model[mask_logg], mass_model[mask_logg],
                              deg=deg[1])

        res_age_fit.append(res_age)
        res_mass_fit.append(res_mass)

        x_fit = np.linspace(2000, up_lim - 2000, 1000)
        y_age_fit = np.polyval(res_age, x_fit)
        y_mass_fit = np.polyval(res_mass, x_fit)

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.scatter(teff_model[mask_logg], age_model[mask_logg], s=5,
                    label='model {}\n logg {}'.format(model, x))
        ax1.plot(x_fit, y_age_fit, 'k-')
        ax1.set_xlabel('Teff')
        ax1.set_ylabel('Age')
        ax1.set_xlim(1000, up_lim)
        ax1.legend()

        ax2.scatter(teff_model[mask_logg], mass_model[mask_logg], s=5)
        ax2.plot(x_fit, y_mass_fit, 'k-')
        ax2.set_xlabel('Teff')
        ax2.set_ylabel('Mass')
        plt.tight_layout()
        name = 'model_' + model + '_logg_' + str(x) + '.png'
        plt.savefig(
            '/Users/rociokiman/Documents/wdwarfdate/Models/cooling_models/Fits/'
            + name,
            tight_layout=True,
            dpi=300)
        if plot_fit == True:
            plt.show()
        plt.close(f)

    return np.array(logg_model_unique), np.array(res_age_fit), np.array(
        res_mass_fit)


def fit_eep_model(feh, vvcrit, deg=8):
    path = '/Users/rociokiman/Documents/wdwarfdate/Models/MIST/MIST_v1.2_feh_'
    table_model = Table.read(path + feh + '_afe_p0.0_vvcrit' + vvcrit
                             + '_EEPS_sum.csv')

    initial_mass = table_model['initial_mass']
    ms_age = table_model['ms_age']

    res_age = np.polyfit(np.log10(initial_mass), np.log10(ms_age), deg=deg)

    x_fit = np.linspace(-1, 2.5, 100)
    y_fit = np.polyval(res_age, x_fit)
    label = 'MIST_v1.2_feh_' + feh + '_afe_p0.0_vvcrit' + vvcrit + '_EEPS'

    f = plt.figure()
    plt.plot(np.log10(initial_mass), np.log10(ms_age), '.',
             label=label)
    plt.plot(x_fit, y_fit, '-k')
    plt.legend()
    plt.xlabel(r'$log_{10}$(initial mass)')
    plt.ylabel(r'$log_{10}$(ms age)')
    plt.savefig(
        '/Users/rociokiman/Documents/wdwarfdate/Models/MIST/Fits/fit_'
        + label + '.png',
        tight_layout=True, dpi=300)
    plt.close(f)
    return res_age

# fit_eep_model(feh = 'p0.00',vvcrit = '0.0')
# fit_eep_model(feh = 'p0.00',vvcrit = '0.4')
# fit_eep_model(feh = 'm4.00',vvcrit = '0.0')
# fit_eep_model(feh = 'm4.00',vvcrit = '0.4')
# fit_eep_model(feh = 'p0.50',vvcrit = '0.0',deg=4)
# fit_eep_model(feh = 'p0.50',vvcrit = '0.4',deg=4)
