# Running the model from the command line:
# python3 <this_file>.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma
import elfi
import scipy.stats
import os

cwd = os.getcwd()
cwd = cwd[0:len(cwd) - len("/cluster")]
print(f"Current working directory: {cwd}")

# TODO: reading simulation parameters from the command line to this script
# TODO: test that this works

## Simulation options ##

use_obs_data = False # use observed or simulated data
is_agg = False # aggregate data?
reparam = False # reparametrized or not?
clade = "A" # clade of interest
obs_data = "NORM" # NORM, BSAC

theta_bsi = 0.3 # proportion of population interest (age group) with BSI
theta_c = 1 # proportion of colonized


# parameters for simulated data

net_transmission_param = 2
R_param = 5

beta = 0.734
gamma = 0.34

# ELFI-related simulation parameters
elfi.new_model()

prior_type = "gamma" # what prior configuration to use for the parameters of interest?

if reparam:
    par1 = elfi.Prior(scipy.stats.uniform,0.01,20)
    par2 = elfi.Prior(scipy.stats.uniform, 2, 10)
else:
    if prior_type == "gamma_normal":
        par1 = elfi.Prior(scipy.stats.gamma, 1, 0, 1/0.1)
        par2 = elfi.Prior(scipy.stats.normal, 1/30, 0.01)
    elif prior_type == "gamma_constant":
        par1 = elfi.Prior(scipy.stats.gamma, 1, 0, 1/0.1)
        par2 = elfi.Constant(1/30)
    elif prior_type == "uniform":
        par1 = elfi.Prior(scipy.stats.uniform, 0, 10)
        par2 = elfi.Prior(scipy.stats.uniform, 0, 10)
    elif prior_type == "gamma":
        par1 = elfi.Prior(scipy.stats.gamma, 1, 0, 1/0.1)
        par2 = elfi.Prior(scipy.stats.gamma, 1, 0, 1/0.1)    
    else:
        print("Warning! Unknown prior type. Using gamma priors.")
        par1 = elfi.Prior(scipy.stats.gamma, 1, 0, 1/0.1)
        par2 = elfi.Prior(scipy.stats.gamma, 1, 0, 1/0.1)
        
        
bs = 10 # batch size
n_iters = 2000 # elfi.sample input


# create an identifier for figures/models:
agg = ""
if is_agg:
    agg == "agg_"
figtag = f"{obs_data}_{clade}_{agg}" # an identifier for saved figures and files

if reparam:
    figtag += "_reparam"
    if not use_obs_data:
        figtag += f"_sim-{net_transmission_rate_param}-{R_param}"
else:
    figtag += f"_{prior_type}"
    if not use_obs_data:
        figtag += f"_sim-{beta}-{gamma}"    
        

        
## Loading the data ##

from load_data import *

if obs_data == "NORM":
    bsi_obs_data = get_obs_BSI(df = norm_data, clade = clade, is_prop = True)
else:
    bsi_obs_data = get_obs_BSI(df = bsac_data, clade = clade, is_prop = True)
print(bsi_obs_data)

n_years = max(bsi_obs_data.index) - min(bsi_obs_data.index) # years of interest
n_weeks = (n_years + 1)*52 # weeks of interest
print(f'Number of weeks to simulate: {n_weeks}')

pop_size = 100000
print(f'Population size: {pop_size}')

## Observational model functions ##

import importlib
import BSI_functions

importlib.reload(BSI_functions) # for changes in the file to take effect

from BSI_functions import * # includes SIR_and_BSI_simulator

## SIR (colonisation simulation) functions ##

import SIR_functions
importlib.reload(SIR_functions)
from SIR_functions import *


## ELFI functions ###

def BSI_max_t(y):
    # time to peak/maximum number of bsi cases
    # shaped (batch_size, n_obs)
    
    return np.argmax(y, axis = 1)

def BSI_max(y):
    # maximum number of BSI cases
    
    max_bsi = np.max(y[:,], axis = 1)
    
    return max_bsi

# A simple SIR with elfi

#elfi.new_model()

is_p = True

# Actual data:
#bsi_obs = np.asarray(get_obs_BSI(norm_data, clade = clade)).reshape(1,-1)
#print(bsi_obs)
bsi_pars = {"or_data": or_data, "clade": clade, "dataset": obs_data, "theta_c":theta_c, "theta_bsi":theta_bsi}

if use_obs_data:
    bsi_obs = bsi_obs_data.reshape(1,-1)
else:
    if reparam:
        bsi_obs = SIR_and_BSI_simulator(net_transmission_param, R_param, nt = n_weeks, N = pop_size, bsi_pars = bsi_pars, is_prop = is_p, is_agg = is_agg, time_period = 52, reparam = True, batch_size = 1, random_state = None)
    else:
        bsi_obs = SIR_and_BSI_simulator(beta, gamma, nt = n_weeks, N = pop_size, bsi_pars = bsi_pars, is_prop = is_p, is_agg = is_agg, time_period = 52, reparam = False, batch_size = 1, random_state = None)

#bsi_obs = (aggregate_BSI(bsi_obs, nan_locations), aggregate_BSI(bsi_obs, nan_locations), aggregate_BSI(bsi_obs, nan_locations))
print(bsi_obs.shape)
print(bsi_obs.ndim)


#beta = elfi.Prior(scipy.stats.uniform, 0, 1)
#gamma = elfi.Prior(scipy.stats.uniform, 0, 1)
#gamma = elfi.Constant(1/30)

# Clancy et al: uninformative gamma priors for beta and gamma
# Lintusaari et al 2016


#beta = elfi.Prior(scipy.stats.uniform, 0, 1)
#gamma = elfi.Prior(scipy.stats.norm,1/30,0.01)
#gamma = elfi.Prior(scipy.stats.uniform, 0, 1)

nt = elfi.Constant(n_weeks)
N = elfi.Constant(pop_size)
is_prop = elfi.Constant(is_p)
is_agg = elfi.Constant(is_agg)
time_period = elfi.Constant(52)
is_reparam = elfi.Constant(reparam)

bsi_pars = elfi.Constant(bsi_pars)
SIR_col = elfi.Simulator(SIR_and_BSI_simulator, par1, par2, nt, N, bsi_pars, is_prop, is_agg, time_period, is_reparam, observed = bsi_obs)

S1 = elfi.Summary(BSI_max_t, SIR_col)
S2 = elfi.Summary(BSI_max, SIR_col)
#S3 = elfi.Summary(I_var_bsi, SIR)

d = elfi.Distance('euclidean', S1, S2)

elfi.draw(d)

elfi.set_client('multiprocessing') # parallellization

### Simulation ###
arraypool = elfi.ArrayPool(['par1', 'par2', 'SIR', 'd'], name = figtag, prefix = f"elfi_output/") # saves the simulated output

smc = elfi.AdaptiveThresholdSMC(d, batch_size=bs, seed=2, q_threshold=0.995, pool = arraypool)
smc_samples = smc.sample(n_iters, max_iter=10)


# display results

#print("R0:", smc_samples.samples['par1'].mean()/smc_samples.samples['par2'].mean())
print("Means of the parameters:", smc_samples.samples['par1'].mean(), smc_samples.samples['par2'].mean())
smc_samples.plot_pairs()
plt.savefig(f"res/{figtag}_pairs.pdf") #plt.show()


if not reparam:
    plt.hist(smc_samples.samples['par1']/smc_samples.samples['par2'])
    plt.title("RO = beta/gamma")
    plt.show()

arraypool.save()

print("Done!")