# BSI_functions.py: Functions related to the observation model

import numpy as np
import scipy.stats as ss

print("Loading BSI_functions.py")
    
def get_OR_hat_pars(or_data, clade = "A"):
    """Get the odds ratio (mean and standard deviation) from a data frame

    Args:
        or_data (Pandas DataFrame): contains odds ratios of invasiveness for different clades and data collections.
        clade (string): 'A' or 'C2', to select the clade of interest.

    Returns:
        or_mu (float): mean estimate of the odds ratio of invasiveness
        or_sd (float): standard deviation calculated from 95% CI
    """
    
    dataset = "BSAC2"
    if clade in ["B", "B_349", "B_non_349"]:
        df = or_data[or_data["Label"] == f'B ({dataset})']
    else:
        df = or_data[or_data["Label"] == f'{clade} ({dataset})']
    or_mu = df["OR"].values[0]
    
    # Assuming that OR is normally distributed and that the CIs (mu +-2sd) are 95%, then the sd is approximately:
    or_sd = (df["upper"].values[0] - df["lower"].values[0])/4#(2*z*n_sqrt)
    
    return or_mu, or_sd, df["lower"].values[0], df["upper"].values[0]
    
def col_to_BSI(SIR, mu_OR, theta_c = 1, theta_bsi = 0.001, N=None, poisson = True):
    """The observation model translating colonisations to BSI incidence. 

    Args:
        SIR (tuple of numpy arrays): output of SIR model, containing time series counts for the S, I and R compartments.
        mu_OR: mean estimate of the odds ratio of invasiveness of a clade
        theta_c (int or float): proportion of population colonized by any E.coli. Assumed 1.
        theta_bsi (float): proportion of population with BSI
        N (int): population size
        poisson (bool): wether to include the Poisson model

    Returns:
        theta_bsi_a_hat (numpy array): BSI incidence corresponding to the given colonisation.
    """
    
    # Colonized population:
    theta_c_a_counts = SIR[1] # colonisation stored in the I compartment
    theta_c_counts = theta_c * N 
    theta_bsi_counts = N * theta_bsi

    theta_c_0_counts = theta_c_counts - theta_c_a_counts
    #rho_c_C_c = mu_OR.reshape(-1,1) * theta_c_a_counts
    rho_c_C_c = mu_OR * theta_c_a_counts
    if poisson:
        theta_bsi_a_hat = ss.poisson.rvs(0.1 + theta_bsi_counts * rho_c_C_c / (theta_c_0_counts + rho_c_C_c))
    else:
        theta_bsi_a_hat = theta_bsi_counts * rho_c_C_c / (theta_c_0_counts + rho_c_C_c)

    return theta_bsi_a_hat


def sync_timewindow(bsi_obs, Dt, n_years, time_period = 52, batch_size = 1, random_state = None):
    """Expanding the observation time window from n_years to Dt + n_years. This way, colonisation can be simulated before the observation period.
    """

    if bsi_obs.ndim == 1:
        bsi_obs = bsi_obs.reshape(1, bsi_obs.shape[0])
        
    data_in_window = np.zeros((bsi_obs.shape[0], n_years * time_period))
    for i in range(0, bsi_obs.shape[0]):
        obs_window = (np.arange(n_years * 52) + Dt[i]).astype(int)
        data_in_window[i,:] = bsi_obs[i, obs_window]

    return data_in_window


def sum_over_bsi(bsi_obs, time_period = 52, batch_size = 1, random_state = None):
    """Sum weekly BSI incidence to yearly BSI incidence (as we have yearly observed data). Take a sum over every ith week in bsi_obs (from i to i + time_period, where i is the current week).

    Args:
        bsi_obs (numpy array): weekly BSI incidence
        time_period (int): time period to sum over, by default 52 weeks (1 year).
        batch_size (int): parameter related to ELFI.
        random_state (int): parameter related to ELFI.

    Returns:
        agg_bsi (numpy array): yearly BSI incidence
    """
    
    n_years = len(bsi_obs[0]) // time_period

    agg_bsi = []
    for i in range(0, n_years):
        start = i * time_period # which week to start summing at
        end = (i + 1) * time_period # next week
        bsi_obs_yearly = bsi_obs[:,start:end]
        agg_bsi.append(np.sum(bsi_obs_yearly[:,], axis = 1))

    agg_bsi = np.asarray(agg_bsi).transpose()   

    return agg_bsi
