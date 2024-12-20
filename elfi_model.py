# elfi_model.py: This file contains the ELFI implementation of the model

# Imports:
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import graphviz # conda install python-graphviz
import elfi
import importlib

# Load functions for the observation model
import BSI_functions
importlib.reload(BSI_functions) # for changes to take effect
from BSI_functions import get_OR_hat_pars, col_to_BSI, sync_timewindow, sum_over_bsi

# Load functions for the SIR model
import SIR_functions
importlib.reload(SIR_functions) # for changes to take effect
from SIR_functions import *

# Load data and parameters
import load_data
importlib.reload(load_data)
from load_data import * # import data: odds ratios, BSI... # Assuming that data has been loaded!

# Simulation parameters: determines clade, number of weeks to simulate etc.
import sim_params
importlib.reload(sim_params)
from sim_params import *
# additional parameters
no_conv = True # include convolution or not
no_Dt = False # including Dt or not
remove_C2_first = True # remove missing observations from beginning of C2 obs data
print(f"Clade: {clade}") # display the clade of interest

### Utility functions ###
# Summaries, distance metrics etc.

def custom_log(S):
    return np.log(S + 0.5)
    
# Summaries
def BSI_max_t(y):
    # time to maximum number of bsi cases
    return np.argmax(y[:,], axis = 1) #*1000 # ELFI # +1

def BSI_max(y):
    # maximum number of BSI cases
    return np.max(y[:,], axis = 1) # ELFI

def BSI_t0(y):
    # t0
    return y[:,0]

## Custom prior functions for beta, gamma parametrization ##

# Define prior t1 as in Marin et al., 2012 with t1 in range [-b, b]
class CustomPrior_gamma(elfi.Distribution):

    @classmethod
    def rvs(cls, loc, scale, size=1, random_state=None):
        u = scipy.stats.uniform.rvs(loc=0.0, scale=0.5, size=size, random_state=random_state)
        t1 = (1 - np.sqrt(2. * u)) * scale
        return t1 + loc

    @classmethod
    def pdf(cls, x, loc, scale):
        p = ss.uniform.pdf(1 / 2 * (1 - (x - loc) / scale) ** 2, loc=0.0, scale=0.5)
        p = p * np.abs( - 1 / scale + (x - loc)/(scale ** 2))
        return p


class CustomPrior_beta(elfi.Distribution):
    @classmethod
    def rvs(cls, g, loc, scale, size=1, random_state=None):
        t2 = ss.uniform.rvs(loc=g, scale=scale - (g - loc), size=size, random_state=random_state)
        return t2

    @classmethod
    def pdf(cls, x, g, loc, scale):
        return ss.uniform.pdf(x, loc=g, scale=scale - (g - loc))

### Loading the data ###

bsi_obs_data = bsi_obs = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = pop_size, remove_C2_first = remove_C2_first)

if remove_C2_first:
    if clade == "C2":
        n_weeks = n_weeks - 3*52
        n_years = n_years - 3
print(f'Number of weeks to simulate: {n_weeks}')

print(f'Population size: {pop_size}')

# Elfi model with SIR and obs model in separate nodes

m = elfi.new_model()
elfi.set_client('native') 
#elfi.set_client('multiprocessing') # breaks

bs = 10
random_state = None
bsi_obs = np.array([bsi_obs_data])

mu_OR, sd_OR = get_OR_hat_pars(or_data, clade = clade)
print(mu_OR)
bsi_pars = {"or_data": or_data, "clade": clade, "theta_c":theta_c, "theta_bsi":theta_bsi} # assume load_data loads or_data, norm_data and bsac_data
sim_pars = {"n_weeks": n_weeks, "pop_size": pop_size, "bsi_pars":bsi_pars, "is_prop":is_prop, "is_agg":is_agg,\
            "time_period":time_period, "reparam":reparam, "random_state":random_state}


# 208 [week] (4 years) is upper limit for the mean colonization period
loc = 1. / 52.

# Here 26 [week] (0.5 years) is the mean infectious period
# and the mean recovery rate is 1 / 26 [week]
scale = 1. - loc 

if not reparam:
    par2 = elfi.Prior(CustomPrior_gamma, loc, scale, model = m) # gamma
    par1 = elfi.Prior(CustomPrior_beta, par2, loc, scale, model = m) # beta
else:
    par1 = elfi.Prior(scipy.stats.uniform, 0, 0.1, model = m)
    loc = 0
    scale = 1
    a_trunc = 1.01
    b_trunc = 3.00
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    par2 = elfi.Prior(scipy.stats.truncnorm, a, b, model = m)

# t0: time when the first colonisation occurred, unknown

def unscale_Dt(Dt_scaled, factor):

    return Dt_scaled*factor
    
if not no_Dt:
    Additional_years = 5.
    Dt = elfi.Prior(scipy.stats.uniform, 0, 0.0001*Additional_years * 52, model = m)
    Dt_unscaled = elfi.Operation(unscale_Dt, 1000, Dt, model = m)
    n_sim_years = elfi.Constant(n_years, model = m)
    t_array = np.arange((n_years +  Additional_years) * 52)
else:
    t_array = np.arange((n_years) * 52)
    
I0 = elfi.Constant(1.0, model = m)
#is_prop = elfi.Constant(is_prop, model = m)
#reparam_node = elfi.Constant(reparam, model = m)
df = or_data[or_data["Label"] == f'{clade} (BSAC2)']
#OR_hat = elfi.Constant(df["OR"].values[0])
#nt = elfi.Constant(n_weeks, model = m)
N = elfi.Constant(pop_size, model = m)
theta_c = elfi.Constant(theta_c, model = m)
theta_bsi = elfi.Constant(theta_bsi, model = m)

mu_OR = elfi.Constant(mu_OR)

if simulator_model == "SIR":
    SIRsim = elfi.Operation(SIR, par1, par2, I0, t_array, N, reparam, model = m)
    BSI = elfi.Operation(col_to_BSI, SIRsim, mu_OR, theta_c, theta_bsi, np.array(pop_size), model = m)
else:
    print(f"{simulator_model} is not a valid simulator model!")

time_period = elfi.Constant(52, model = m)

if not no_Dt: # include Dt
    synced_data = elfi.Operation(sync_timewindow, BSI, Dt_unscaled, n_sim_years, model = m) # no conv, Dt
    yearly_BSI = elfi.Simulator(sum_over_bsi, synced_data, time_period, model = m, observed = bsi_obs)
else: # do not include Dt
    yearly_BSI = elfi.Simulator(sum_over_bsi, BSI, time_period, model = m, observed = bsi_obs) # no conv, no Dt

# max, max_t, BSI at t0

log_BSI = elfi.Summary(custom_log, yearly_BSI, model = m)
log_BSI_max = elfi.Summary(BSI_max, log_BSI, model = m)
log_BSI_t0 = elfi.Summary(BSI_t0, log_BSI, model = m)
BSI_max_t_summary = elfi.Summary(BSI_max_t, yearly_BSI, model = m)

d = elfi.Distance("euclidean", log_BSI_max, BSI_max_t_summary, log_BSI_t0, model = m)

print("Model imported")
elfi.draw(m)