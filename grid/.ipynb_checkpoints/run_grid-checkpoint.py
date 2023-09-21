import numpy as np
import matplotlib.pyplot as plt
import importlib
import datetime
import os


print(os.getcwd())
os.chdir('/u/50/ojalaf2/unix/Dropbox (Aalto)/Ecoli')
#os.chdir('/scratch/work/ojalaf2/Ecoli')
print(os.getcwd())

import sys
sys.path.append(os.getcwd()) # fixes a ModuleNotFoundError when importing cluster.scripts.load_data



from cluster.scripts.load_data import * # import data: odds ratios, BSI...
import cluster.scripts.BSI_functions
importlib.reload(cluster.scripts.BSI_functions) # for changes to take effect

from cluster.scripts.BSI_functions import *



use_incidence = True # Change to False if you wish to use the proportion of isolates instead

# Read simulation variables from a file

import grid_params
importlib.reload(grid_params)
from grid_params import *

import grid_functions
importlib.reload(grid_functions)
from grid_functions import *

res_id = r_id #+ "_" + datetime.date.today().strftime("%d-%m-%Y")

#res_root = "res/sim_res/"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_directory = f"res/sim_res/{res_id}_{timestamp}"
os.mkdir(output_directory)


# Calculate the distance between simulated and observed sequences in a grid

bsi_pars = {"or_data": or_data, "clade": clade, "dataset": obs_data, "theta_c":theta_c, "theta_bsi":theta_bsi, "include_I0":include_I0} # assume load_data loads or_data, norm_data and bsac_data
sim_pars = {"n_weeks": n_weeks, "pop_size": pop_size, "bsi_pars":bsi_pars, "is_prop":is_prop, "is_agg":is_agg,\
            "time_period":time_period, "reparam":reparam, "batch_size":batch_size, "random_state":random_state}

# Save the parameters of this specific run:
with open(os.path.join(output_directory, "sim_params.txt"), "w") as f:
    for key, value in sim_pars.items():
        if key != "bsi_pars":
            f.write(f"{key}: {value}\n")
    f.write(f"n_grid: {n_grid}\n")
    f.write(f"output_directory: {output_directory}\n")
    for key, value in bsi_pars.items():
        if key != "or_data":
            f.write(f"{key}: {value}\n")


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
    plt.savefig(output_directory + "synthetic_BSI_obs_" + res_id + ".pdf", format="pdf", bbox_inches="tight")
    #plt.show()
    
    theta_bsi_a_0 = bsi_obs[0][0]
    
else: # Use real data
    print(f"Using real data. Dataset: {obs_data}, clade: {clade}")
    
    if use_incidence:
        bsi_obs = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = pop_size)
        bsi_obs = bsi_obs.values
    else:
        if obs_data == "NORM":
            bsi_obs = get_obs_BSI(df = norm_data, clade = clade, is_prop = is_prop)
        else:
            bsi_obs = get_obs_BSI(df = bsac_data, clade = clade, is_prop = is_prop)

    plt.plot(bsi_obs)
    plt.title(f"Real BSI clade: {clade}, dataset: {obs_data}")
    plt.savefig(output_directory + "/real_BSI_obs_" + res_id + ".pdf", format="pdf", bbox_inches="tight")
    
    theta_bsi_a_0 = bsi_obs[0] # bsi_obs.iloc[0]


# Run the simulation

if reparam:
    pairs = get_nt_R_pairs(n_grid, n_grid)
else:
    pairs = get_valid_beta_gamma_pairs(n_grid, n_grid)
    
dists, summary_dists = get_distance_points(pairs, bsi_obs, sim_pars, [BSI_1, BSI_2, BSI_3, BSI_4, BSI_5, BSI_6, BSI_7, BSI_8, BSI_9, BSI_10, BSI_11, BSI_12, BSI_13, BSI_14, BSI_max_t]) #[BSI_max, BSI_max_t], [BSI_vector, BSI_max_t, BSI_max]

# Save dists and pairs

#np.save(output_directory + "pairs" + "_" + res_id + ".npy", pairs)
#np.save(output_directory + "dists" + "_" + res_id + ".npy", dists)
#np.save(output_directory + "summary_dists" + "_" + res_id + ".npy", dists)
np.save(output_directory + "/pairs" + ".npy", pairs)
np.save(output_directory + "/dists" + ".npy", dists)
np.save(output_directory + "/summary_dists" + ".npy", summary_dists)

# Most of the visualization of 'dists' is done in a notebook given how many manual tweaks the figures might need (tolerance etc).

# Scatterplot of parameter pairs:

scatter_distance_points(pairs[:,0], pairs[:,1], dists, true_beta = true_par1, true_gamma = true_par2,\
                        save = True, filename = output_directory + "/grid_scatter" + res_id)


# Generate a set of visualizations:

import grid_functions
importlib.reload(grid_functions)
from grid_functions import *

# I0 included:
#visualize_results("res/sim_res/res_test_2023-08-28_09-44-04", 0.3)