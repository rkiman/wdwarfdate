#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import daft
from matplotlib import rc

def make_and_save_pgm():
    rc("font", family="serif", size=10)
    rc("text", usetex=True)
    
    
    pgm = daft.PGM(observed_style="inner")
    
    # Hierarchical parameters.
    pgm.add_node("sigma", r"$\sigma _m$", 0.2, 3.5, fixed=True)
    pgm.add_node("prior_ms", r"", 1.4, 6.7, fixed=True)
    pgm.add_node("prior_cool", r"", 2.4, 6.7, fixed=True)
    
    # Latent variable.
    pgm.add_node("msage", r"$t_{ms_n}$", 1.4, 6)
    pgm.add_node("coolingage", r"$t_{cool_n}$", 2.4, 6)
    pgm.add_node("totalage", r"$t_{tot_n}$", 1.9, 5.3)
    pgm.add_node("initialmass", r"$m_{i_n}$", 1.4, 5)
    pgm.add_node("deltam", r"$\Delta_{m_n}$", 0.9, 3.5)
    pgm.add_node("finalmass_hat", r"$\hat{m_{f_n}}$", 1.4, 4)
    pgm.add_node("finalmass", r"$m_{f_n}$", 1.4, 3)
    pgm.add_node("teff", r"$T_{eff_n}$", 1.4, 2)
    pgm.add_node("logg", r"$\log g_n$", 2.4, 2)
    
    # Data.
    pgm.add_node("teff_obs", r"$T_{eff_n}$", 1.4, 1, observed=True)
    pgm.add_node("logg_obs", r"$\log g_n$", 2.4, 1, observed=True)
    
    # Add in the edges.
    pgm.add_edge("prior_ms", "msage")
    pgm.add_edge("prior_cool", "coolingage")
    pgm.add_edge("msage", "totalage")
    pgm.add_edge("coolingage", "totalage")
    pgm.add_edge("msage", "initialmass")
    pgm.add_edge("initialmass", "finalmass_hat")
    pgm.add_edge("sigma", "deltam")
    pgm.add_edge("deltam", "finalmass")
    pgm.add_edge("finalmass_hat", "finalmass")
    pgm.add_edge("finalmass", "teff")
    pgm.add_edge("teff", "teff_obs")
    pgm.add_edge("coolingage", "logg")
    pgm.add_edge("coolingage", "teff")
    pgm.add_edge("finalmass", "logg")
    pgm.add_edge("logg", "logg_obs")
    
    # And a plate.
    pgm.add_plate([0.5, 0.5, 2.5, 6], label=r"$n = 1, \ldots, N$", shift=-0.1)
    
    # Render and save.
    pgm.render()
    pgm.savefig("../model.png", dpi=300)


#Leave this commented so it doesn't make a plot everytime I import the package
#make_and_save_pgm()