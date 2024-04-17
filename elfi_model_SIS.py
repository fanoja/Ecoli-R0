# Imports:
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import graphviz # conda install python-graphviz
import elfi
import importlib

import cluster.scripts.BSI_functions
importlib.reload(cluster.scripts.BSI_functions) # for changes to take effect
from cluster.scripts.BSI_functions import *

import cluster.scripts.SIR_functions
importlib.reload(cluster.scripts.SIR_functions) # for changes to take effect
from cluster.scripts.SIR_functions import *


# Load data and parameters:
from cluster.scripts.load_data import * # import data: odds ratios, BSI... # Assuming that data has been loaded!
import grid_params
importlib.reload(grid_params)
from grid_params import * # Load parameters from grid_params.py

print("Grid parameters:")
print(f"Clade: {clade}")
# Specify parameters:

print("NOTE: Using SIS model.")


# Custom prior functions:

class CustomPrior_gamma(elfi.Distribution):
    def rvs(loc, scale, size=1, random_state=None):
        gamma = scipy.stats.norm.rvs(loc=loc, scale=scale, size=size, random_state=random_state)
        return gamma

class CustomPrior_beta(elfi.Distribution):
    def rvs(gamma, min_R0, max_R0, size=1, random_state=None):
        #u = scipy.stats.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        len_gamma = gamma.shape[0]
        beta = np.zeros((len_gamma,))
        for i in range(0, len_gamma):
            g = gamma[:,][i]
            beta[:,][i] = g*scipy.stats.uniform.rvs(loc = min_R0, scale = max_R0, size=1, random_state=random_state)
            
        #print(gamma.shape)
        #beta = scipy.stats.uniform.rvs(loc = gamma*min_R0, scale = gamma*max_R0, size=size, random_state=random_state)   
        return beta

# Specifying data-related parameters:
#bsi_obs_data = get_obs_BSI(df = norm_data, clade = "A", is_prop = True)
bsi_obs_data = get_obs_BSI(df = norm_data, clade = clade, is_prop = is_prop)
bsi_obs_data = bsi_obs = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = pop_size)

print(f'Number of weeks to simulate: {n_weeks}')

print(f'Population size: {pop_size}')

print(bsi_obs_data)

# Elfi model with SIR and obs model in separate nodes

m = elfi.new_model()
elfi.set_client('native') # parallellization

bs = 10
random_state = None

#SIR_obs = SIR(np.array([0.224]), np.array([0.0456]), nt = n_weeks, N = pop_size, batch_size = 1)
#OR_hat = get_OR_hat(or_data, clade = "A", dataset = "NORM")
#bsi_obs = col_to_BSI(SIR_obs, OR_hat = OR_hat, is_prop = True)#.flatten()
bsi_obs = np.array([bsi_obs_data])

obsS1 = BSI_max_t(bsi_obs)
obsS2 = BSI_max(bsi_obs)
obs_summaries = (obsS1, obsS2)

mu_OR, sd_OR = get_OR_hat_pars(or_data, clade = clade, dataset = obs_data)
bsi_pars = {"or_data": or_data, "clade": clade, "dataset": obs_data, "theta_c":theta_c, "theta_bsi":theta_bsi, "include_I0":include_I0} # assume load_data loads or_data, norm_data and bsac_data
sim_pars = {"n_weeks": n_weeks, "pop_size": pop_size, "bsi_pars":bsi_pars, "is_prop":is_prop, "is_agg":is_agg,\
            "time_period":time_period, "reparam":reparam, "batch_size":bs, "random_state":random_state}

# Clancy et al: uninformative gamma priors for beta and gamma
# TODO: restrict these to be over 0 always
#beta = elfi.Prior(scipy.stats.gamma, 1, 0, 10) # transmission coefficient

#beta = elfi.Prior(scipy.stats.uniform, 0.0015, 1, model = m) # Given the gamma prior and the assumption that R is between 1.5 and 10, this should be a reasonable range for beta

#beta = elfi.Prior(scipy.stats.uniform, 0, 1)
#gamma = elfi.Prior(scipy.stats.gamma, 1, 0, 100)

if not reparam:
    
    #gamma = elfi.Prior(scipy.stats.uniform,0, 2, model = m)
    gamma = elfi.Prior(scipy.stats.gamma(a = 1.5,  scale = 1/6), model = m)

    """
    if clade == "A":
        #gamma = elfi.Prior(scipy.stats.norm,1/30,0.01, model = m) # recovery rate
        #gamma = elfi.Prior(scipy.stats.gamma(a = 1.5,  scale = 1/6), model = m)
        gamma = elfi.Prior(scipy.stats.norm, 7/30, 7*0.01, model = m)
    else:
        #gamma = elfi.Prior(scipy.stats.uniform, 0, 100, model = m) # Test a uniform prior for gamma, as in grid.
        #gamma = elfi.Prior(scipy.stats.norm,1/30,0.01, model = m)
        #gamma = elfi.Prior(scipy.stats.gamma(a = 1.5,  scale = 1/6), model = m)
        gamma = elfi.Prior(scipy.stats.norm, 7/30, 7*0.01, model = m)
        
    """
        
    beta = elfi.Prior(CustomPrior_beta, gamma, 1.01, 5, model = m) 
    #beta = elfi.Prior(scipy.stats.uniform, 0, 1)

# Reparametrized version:
if reparam:
    #net_transmission = elfi.Prior(scipy.stats.uniform, 0.3, 1, model = m)
    #net_transmission = elfi.Prior(scipy.stats.uniform, 0.03, 0.5, model = m) # Leads to different SIR curves -> good, But maybe too restrictive?
    # In grid net_transmission prior is: [0.001, 0.8]
    #net_transmission = elfi.Prior(scipy.stats.uniform, 0.001, 0.799, model = m) # 0.001 to 0.8 (same as grid)
    net_transmission = elfi.Prior(scipy.stats.uniform, 0, 0.3, model = m)
    #net_transmission_log = elfi.Operation(np.log, net_transmission, model = m)
    #net_transmission = elfi.Prior(scipy.stats.beta, 1.5, 8, model = m)
    R = elfi.Prior(scipy.stats.uniform, 1.01, 4.99, model = m)
    #R_log = elfi.Operation(np.log, R, model = m)

nt = elfi.Constant(n_weeks, model = m)
N = elfi.Constant(pop_size, model = m)

#mu_OR = elfi.Constant(mu_OR, model = m) # exp for lognormal
#sd_OR = elfi.Constant(sd_OR, model = m)
#OR_hat = elfi.RandomVariable(scipy.stats.norm, mu_OR, sd_OR, model = m)


df = or_data[or_data["Label"] == f'{clade} (BSAC2)']
OR_hat = elfi.Constant(df["OR"].values[0])
#OR_hat = elfi.Prior(scipy.stats.norm, mu_OR, sd_OR, model = m)
#OR_hat = elfi.Prior(scipy.stats.lognorm, scale = mu_OR, s = sd_OR, model = m)


I0 = elfi.Constant(None, model = m)
is_prop = elfi.Constant(is_prop, model = m)
reparam_node = elfi.Constant(reparam, model = m)

# This node outputs SIR simulations, but the observed data is BSI.


if reparam:
    SISsim = elfi.Operation(SIS, net_transmission, R, nt, N,I0, reparam_node, is_prop, model = m)
else:
    SISsim = elfi.Operation(SIS, beta, gamma, nt, N, I0, reparam_node, is_prop, model = m)

# SIRsim = elfi.Simulator(SIR, beta, gamma, nt, N, observed = bsi_obs) # PROBLEM: The observations do not match this node. 

#nSIR = elfi.Operation(prop_to_nSIR, SIRsim, N)

def scale_theta_BSI(theta_BSI, scaling_factor):
    
    return scaling_factor*theta_BSI

theta_c = elfi.Constant(theta_c, model = m)
#l = elfi.Prior(scipy.stats.uniform,0, 2, model = m) # Scaling factor for theta_bsi, was 0, 0.1 for best fit.
l = elfi.Constant(1, model = m)
theta_bsi_unscaled = elfi.Constant(theta_bsi, model = m)
theta_bsi = elfi.Operation(scale_theta_BSI, theta_bsi_unscaled, l, model = m)

#theta_bsi = elfi.Prior(scipy.stats.uniform, 0, 1.9e-5, model = m) # 1.9e-5

#alpha = elfi.Prior(scipy.stats.uniform, 0, 1, model = m)

alpha = elfi.Constant(1.0, model = m)

#is_prop = elfi.Constant(False, model = m)

# elfi.Simulator(col_to_BSI, SIRsim, OR_hat, theta_c, theta_bsi, is_prop, observed = bsi_obs)

BSI = elfi.Operation(col_to_BSI, SISsim, OR_hat, theta_c, theta_bsi, is_prop, model = m)
    
time_period = elfi.Constant(52, model = m)
sum_BSI = elfi.Operation(sum_over_bsi, BSI, time_period, model = m)
#sum_BSI = elfi.Simulator(sum_over_bsi, BSI, time_period, observed = bsi_obs, model = m)
                         
conv_BSI = elfi.Simulator(exp_smoother, sum_BSI, alpha, observed = bsi_obs, model = m)
#conv_BSI = elfi.Operation(exp_smoother, sum_BSI, alpha, model = m)

def BSI_max_partial(y, p = 0.95):
# Find a partial maximum value. p = percent of the maximum.
# Finds the closest value to p*maximum

    max_bsi_p = np.zeros(y.shape[0])
    p_max = np.max(y[:,], axis = 1)*p
    for b in range(0, y.shape[0]):
        pm = p_max[b]
        max_bsi_p[b] = y[b,np.abs(y[b,]-pm).argmin()]

    return np.log(max_bsi_p + 1)#.reshape(-1,1).transpose()

def BSI_max_partial_t(y, p = 0.95):
    # Find a partial maximum value. p = percent of the maximum.
    # Finds the closest value to p*maximum

    max_bsi_p = np.zeros(y.shape[0])
    p_max = np.max(y[:,], axis = 1)*p
    for b in range(0, y.shape[0]):
        pm = p_max[b]
        max_bsi_p[b] = np.abs(y[b,]-pm).argmin()

    return max_bsi_p#.reshape(-1,1).transpose()

def mean_cumulative_diff(y):
    # Summary: mean difference in log cumulative cases
    cy = np.cumsum(y[:,], axis = 1)
    return np.mean(np.diff(np.log(cy + 1))[:,], axis = 1)
    
# Regression-based summaries:
def fit_linreg(y, plot = False, cumulative = False):
    
    from sklearn import linear_model
    
    if cumulative:
        y = np.cumsum(y, axis = 1)
    
    log_data = np.log(y + 1)

    bs = y.shape[0]
    slopes = np.zeros(bs)
    ints = np.zeros(bs)

    for b in range(0, bs):
        reg = linear_model.LinearRegression()

        x = np.arange(0, len(y[b]), 1).reshape(-1, 1)
        y_train = log_data[b] 
        reg.fit(x, y_train)

        if plot:
            y_pred = reg.predict(x)

            plt.plot(x, y_pred)
            plt.scatter(x,y_train)
            plt.title("Regression line on log(x + 1) data")
            plt.show()

        ints[b] = reg.intercept_
        slopes[b] = reg.coef_

        
    return slopes, ints

def linreg_slope(y):
    
    return fit_linreg(y)[0]

def linreg_intercept(y):
    
    return fit_linreg(y)[1]


"""
S1 = elfi.Summary(BSI_max_partial, sum_BSI, model = m)
S2 = elfi.Summary(BSI_max_partial_t, sum_BSI, model = m)
S3 = elfi.Summary(BSI_k0, sum_BSI, model = m)
"""

"""
S1 = elfi.Summary(BSI_max_partial, conv_BSI, model = m)
S2 = elfi.Summary(BSI_max_partial_t, conv_BSI, model = m)
S3 = elfi.Summary(BSI_k0, conv_BSI, model = m)
"""

"""
S1 = elfi.Summary(mean_cumulative_diff, conv_BSI, model = m)
S2 = elfi.Summary(BSI_max_partial, conv_BSI, model = m)
S3 = elfi.Summary(BSI_max_partial_t, conv_BSI, model = m)
"""

S1 = elfi.Summary(BSI_1, conv_BSI, model = m)
S2 = elfi.Summary(BSI_2, conv_BSI, model = m)
S3 = elfi.Summary(BSI_3, conv_BSI, model = m)
S4 = elfi.Summary(BSI_4, conv_BSI, model = m)
S5 = elfi.Summary(BSI_5, conv_BSI, model = m)
S6 = elfi.Summary(BSI_6, conv_BSI, model = m)
S7 = elfi.Summary(BSI_7, conv_BSI, model = m)
S8 = elfi.Summary(BSI_8, conv_BSI, model = m)
S9 = elfi.Summary(BSI_9, conv_BSI, model = m)
S10 = elfi.Summary(BSI_10, conv_BSI, model = m)
S11 = elfi.Summary(BSI_11, conv_BSI, model = m)
S12 = elfi.Summary(BSI_12, conv_BSI, model = m)
S13 = elfi.Summary(BSI_13, conv_BSI, model = m)
S14 = elfi.Summary(BSI_14, conv_BSI, model = m)
#linreg = elfi.Simulator(fit_linreg, conv_BSI, observed = bsi_obs, model = m)
#linreg = elfi.Operation(fit_linreg, conv_BSI, model = m)

#S1 = elfi.Summary(linreg_slope, conv_BSI, model = m)
#S2 = elfi.Summary(linreg_intercept, conv_BSI, model = m)


#S1 = elfi.Summary(linreg_slope, conv_BSI, model = m)
#S2 = elfi.Summary(linreg_intercept, conv_BSI, model = m)

d = elfi.Distance('euclidean', S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, model = m)
#d = elfi.Distance('euclidean', S1, S2, S3, model = m)
    
def custom_log(S):
    return np.atleast_2d(np.log(S + 1)).reshape(-1,1)

print("Model imported")
elfi.draw(m)