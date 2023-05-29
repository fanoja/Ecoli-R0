import numpy as np
import matplotlib.pyplot as plt
import importlib
import datetime
import os

print(os.getcwd())
os.chdir('/u/50/ojalaf2/unix/Dropbox (Aalto)/Ecoli/')

from cluster.scripts.load_data import * # import data: odds ratios, BSI...
import cluster.scripts.BSI_functions
importlib.reload(cluster.scripts.BSI_functions) # for changes to take effect

from cluster.scripts.BSI_functions import *

res_root = "res/"
use_incidence = True # Change to False if you wish to use the proportion of isolates instead

# Read simulation variables from a file

import grid_params
importlib.reload(grid_params)
from grid_params import *

import grid_functions
importlib.reload(grid_functions)
from grid_functions import *

# Calculate the distance between simulated and observed sequences in a grid

bsi_pars = {"or_data": or_data, "clade": clade, "dataset": obs_data, "theta_c":theta_c, "theta_bsi":theta_bsi} # assume load_data loads or_data, norm_data and bsac_data
sim_pars = {"n_weeks": n_weeks, "pop_size": pop_size, "bsi_pars":bsi_pars, "is_prop":is_prop, "is_agg":is_agg,\
            "time_period":time_period, "reparam":reparam, "batch_size":batch_size, "random_state":random_state}

res_id = r_id #+ "_" + datetime.date.today().strftime("%d-%m-%Y")

if true_par1 != None:
    print("Using synthetic data.")
    bsi_obs = SIR_and_BSI_simulator(np.array([true_par1]), np.array([true_par2]), nt = n_weeks, N = pop_size,\
                                bsi_pars = bsi_pars, is_prop = is_prop,\
                                is_agg = is_agg, time_period = time_period,\
                                batch_size = batch_size, random_state = random_state)
    
    # Save simulated data figure, both aggregated and non-aggregated
    plt.plot(bsi_obs[0])
    if reparam:
        plt.title(f"BSI clade {clade}, {obs_data}\n Net transmission = {true_par1}, R = {true_par2}")
    else:
        plt.title(f"BSI clade {clade}, {obs_data}\n Beta = {true_par1}, gamma = {true_par2}")
    plt.savefig(res_root + "synthetic_BSI_obs_" + res_id + ".pdf", format="pdf", bbox_inches="tight")
    #plt.show()
    
else: # Use real data
    print(f"Using real data. Dataset: {obs_data}, clade: {clade}")
    
    if use_incidence:
        bsi_obs = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = pop_size)
    else:
        if obs_data == "NORM":
            bsi_obs = get_obs_BSI(df = norm_data, clade = clade, is_prop = is_prop)
        else:
            bsi_obs = get_obs_BSI(df = bsac_data, clade = clade, is_prop = is_prop)

    plt.plot(bsi_obs)
    plt.title(f"Real BSI clade: {clade}, dataset: {obs_data}")
    plt.savefig(res_root + "real_BSI_obs_" + res_id + ".pdf", format="pdf", bbox_inches="tight")
    

# Run the simulation

if reparam:
    pairs = get_nt_R_pairs(n_grid, n_grid)
else:
    pairs = get_valid_beta_gamma_pairs(n_grid, n_grid)
    
dists, summary_dists = get_distance_points(pairs, bsi_obs, sim_pars, [BSI_max, BSI_max_t])

# Save dists and pairs

np.save(res_root + "pairs" + "_" + res_id + ".npy", pairs)
np.save(res_root + "dists" + "_" + res_id + ".npy", dists)
np.save(res_root + "summary_dists" + "_" + res_id + ".npy", dists)

# Most of the visualization of 'dists' is done in a notebook given how many manual tweaks the figures might need (tolerance etc).

# Scatterplot of parameter pairs:

scatter_distance_points(pairs[:,0], pairs[:,1], dists, true_beta = true_par1, true_gamma = true_par2,\
                        save = True, filename = res_root + "grid_scatter" + res_id)