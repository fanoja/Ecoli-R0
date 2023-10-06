import numpy as np
import matplotlib.pyplot as plt
import os
    
print("Loading BSI_functions.py")

grid = False
    
def get_OR_hat_pars(or_data, clade = "A", dataset = "NORM"):
    
    dataset = "BSAC"
    df = or_data[or_data["Label"] == f'{clade} ({dataset})']
    or_mu = df["OR"]
    or_sd = (df["upper"] - df["lower"])/2
    
    return or_mu, or_sd
    
def get_OR_hat(or_data, clade = "A", dataset = "NORM", batch_size = 1, random_state = None):
    # TODO: fix that random state
    or_mu, or_sd = get_OR_hat_pars(or_data, clade = clade, dataset = dataset) 
    
    # using log odds for the normal distribution: 
    #or_mu = np.log(or_mu)
    #or_sd = np.log(or_sd)
    
    #print(or_sd)
    #print(or_mu)
    
    OR_hats = np.empty(batch_size)
    
    #or_sd = 1e12*or_sd
    
    for b in range(0, batch_size):
        OR_hat = np.random.normal(or_mu, or_sd, 1) # laita nollaan jos negatiivinen
        
        #if OR_hat < 0: # causes divide by zero issues in grid
            #OR_hat = 0
        max_iter = 1000
        i = 0

        
        while OR_hat[0] <= 0:
            OR_hat = np.random.normal(or_mu, or_sd, 1)
            i = i + 1
            if i == max_iter:
                break

        #if i > 0:
            #print(f"Iterated OR_hat {i} times due to negativity.")

        if OR_hat < 0:
            print(f"Warning, negative OR_hat after max iterations!")
        
        OR_hats[b] = OR_hat
    
    return OR_hats

def col_to_BSI(SIR, OR_hat, theta_c = 1, theta_bsi = 0.001, is_prop = True, batch_size = 1, random_state = None):
    # SIR: output of the SIR simulator (clade of interest colonization proportion over time)
    # theta_bsi: The proportion of bsi in the entire (colonized) population - from the age distribution.
    # theta_c: The overall proportion of population colonized by E. coli. For simplicity, assume we are only interested in the colonized 
    # population and set theta_c = 1 by default.
    # By changing is_prop = False, can work with counts instead of proportions
    
    #if isinstance(theta_c, elfi.model.elfi_model.Constant):
        #theta_c = theta_c.generate(1)
    #if isinstance(theta_bsi, elfi.model.elfi_model.Constant):
        #theta_bsi = theta_bsi.generate(1)
        
    theta_c_a = SIR[1]
    #print(theta_c_a)
    
    if not is_prop: 
        N = SIR[0][0][0] + 1 # the first entry in S compartment + 1 is the population size.
        #print(f"N: {N}")
        theta_c = N*theta_c 
        theta_bsi = theta_bsi*N
    else:
        if np.min(np.max(SIR[1][:,], axis = 1)) > 1:
            print("Warning! col_to_BSI uses proportions, but SIR seems to use counts.")
    
    bs = theta_c_a.shape[0]
    n_obs = theta_c_a.shape[1] # with batches. 0 = first batch
    #print(f"n_obs: {n_obs}")

    theta_c_0 = theta_c - theta_c_a
    #print(f"theta_c_0: {theta_c_0}")

    #if is_agg:
        #theta_bsi = theta_bsi/52
    #print(OR_hat)
    #print(OR_hat, theta_bsi, theta_c)
    theta_bsi_a_hat = OR_hat.reshape(-1,1)*theta_c_a*theta_bsi/(theta_c_0 + OR_hat.reshape(-1,1)*theta_c_a)
    
    #print(f"Shape of the theta_bsi_a_hat in col_to_BSI {theta_bsi_a_hat.shape}")
    theta_bsi_a_hat = np.maximum(theta_bsi_a_hat, 0) # If elements are below zero, force them to be 0.
    #print(theta_bsi_a_hat)
    return theta_bsi_a_hat


def sum_over_bsi(bsi_obs, time_period = 52, batch_size = 1, random_state = None):
    # Take a sum over every ith week in bsi_obs (from i to i + time_period, where i is the current week)
    # Note: probably not applicable for proportion observations, only counts
    # time_period: time period to sum over. By default 52 (weeks).
    
    n_years = len(bsi_obs[0])//52
    #print(f'Number of years to sum: {n_years}')
    agg_bsi = []
    for i in range(0, n_years):
        start = i*time_period # which week to start summing at
        end = time_period*i + time_period # next week
        bsi_obs_yearly = bsi_obs[:,start:end]
        agg_bsi.append(np.sum(bsi_obs_yearly[:,], axis = 1))
    agg_bsi = np.asarray(agg_bsi).transpose()   
    
    return agg_bsi

def max_bsi_per_year(bsi_obs, time_period = 52):
    # Return the maximum number of BSI cases per yea/r
    
    n_years = len(bsi_obs[0])//52
    agg_bsi = []
    
    for i in range(0, n_years):
        start = i*time_period # which week to start summing at
        end = time_period*i + time_period # next week
        agg_bsi.append(np.max(bsi_obs[:,start:end], axis = 1))
    agg_bsi = np.asarray(agg_bsi).transpose()  
    
    return agg_bsi

def plot_col_to_BSI(SIR, or_data, clade = "A", dataset = "NORM", n_rep = 100, theta_c = 1, theta_bsi = 0.3, is_prop = True, xlabel = "Years"):
    # Plot n_rep repetitions of theta_BSI_clade as "translated" from colonization by clade of interest.
    
    #print(is_prop)
    all_bsi_reps= []
    for i in range(0, n_rep):
        or_hat = get_OR_hat(or_data, clade = clade, dataset = dataset)
        obsBSI = col_to_BSI(SIR, or_hat, theta_c = theta_c, theta_bsi = theta_bsi, is_prop = is_prop)
        #print(obsBSI)
        if i == 0:
            plt.plot(obsBSI[0], color = "lightblue", label = "Theta_bsi_A")
        else:
            plt.plot(obsBSI[0], color = "lightblue")
        all_bsi_reps.append(obsBSI)

    plt.plot(SIR[1][0], color = "red", label = "theta_c_A")
    plt.plot(np.mean(all_bsi_reps, axis = 0)[0], color = "navy", label = "Mean of BSI reps")
    plt.xlabel(xlabel)
    if is_prop:
        plt.title(f"Proportion of BSI and col: Clade {clade}, {dataset}\n theta_c = {theta_c}, theta_bsi = {theta_bsi}")
        plt.ylabel("Proportion")
    else:
        plt.title(f"Number of BSI cases: Clade {clade}, {dataset}")
        plt.ylabel("Count")
    plt.legend()
    plt.show()
    
def plot_BSI(y_bsi):
    
    if len(y_bsi) > 1:
        plt.plot(y_bsi[0][0], label = "BSI associated with the clade of interest")
    else:
        plt.plot(y_bsi[0], label = "BSI associated with the clade of interest")
    plt.legend()
    plt.show()

    
### Combining SIR and the observational model (BSI model) ###
import re
from cluster.scripts.load_data import *


def exp_smoother(bsi, alpha = 0.2, batch_size = 1, random_state = None):
    # Assumes an array bsi shaped (batch_size, n_obs)
    bs = bsi.shape[0] # batch size.
    bsi_filtered = np.zeros((bs, len(bsi[0])))

    for i in range(0, bs):
        x = bsi[i]
        bsi_filtered[i][0] = x[0] # initialize at first value
        for j in range(1, len(x)):
            
            if isinstance(alpha, float): # grid
                bsi_filtered[i][j] = alpha*x[j] + (1-alpha)*bsi_filtered[i][j-1] # Exponential smoothing
            else: # ELFI with batches 
                bsi_filtered[i][j] = alpha[i]*x[j] + (1-alpha[i])*bsi_filtered[i][j-1] # Exponential smoothing
   
    #print(f"Shape of bsi_filtered in exp_smoother: {bsi_filtered.shape}")
    return bsi_filtered # returns an array of shape (bs, n_obs)


def SIR_and_BSI_simulator(par1, par2, nt, N, bsi_pars, alpha = 0.2, is_prop = True, is_agg = False, time_period = 52, reparam = False, has_or_hat = False, manual_or_hat = None, batch_size = 1, random_state = None):
    # A simulator function combining both the SIR simulation and the observational model
    
    cwd = os.getcwd()

    if bool(re.search('cluster', cwd)): # a hack to get these loaded from the main directory vs from cluster/
        from scripts.SIR_functions import SIR, prop_to_nSIR
    else:
        from cluster.scripts.SIR_functions import SIR, prop_to_nSIR

    
    or_data = bsi_pars["or_data"]
    clade = bsi_pars["clade"]
    dataset = bsi_pars["dataset"]
    theta_c = bsi_pars["theta_c"]
    theta_bsi = bsi_pars["theta_bsi"]
    
    # Find I0:
    if dataset == "NORM":
        bsi_obs = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = N)
    else:
        bsi_obs = get_obs_BSI(df = bsac_data, clade = clade, is_prop = is_prop)
    
    if not has_or_hat:
        or_hat = get_OR_hat(or_data = or_data, clade = clade, dataset = dataset, batch_size = batch_size, random_state = random_state)
    else:
        or_hat = manual_or_hat
    
    #or_hat = get_OR_hat(or_data = or_data, clade = clade, dataset = dataset, batch_size = batch_size, random_state = random_state)
    
    if bsi_pars["include_I0"]:
        theta_bsi_a_0 = bsi_obs.iloc[0]/time_period
        I0 = (theta_bsi_a_0*theta_c/(theta_bsi_a_0 + or_hat[0]*theta_bsi - theta_bsi_a_0*or_hat[0]))*N
    else:
        I0 = None
    #print(f"theta_bsi_a_0 is {theta_bsi_a_0}")
    #print(f"I0 is {I0}")
    #print(f"or_hat {or_hat}")
    
    #sim_pars["I0"] = I0
    
    
    # SIR simulator:
    
    SIRsim = SIR(par1, par2, nt = nt, N = N, I0 = I0, reparam = reparam, is_prop = is_prop, batch_size = batch_size, random_state = random_state)
    #print(SIRsim)
    
    #if not is_prop:
        #SIRsim = prop_to_nSIR(SIRsim, N)
        
    # Observational model:
    

    
    #or_hat = get_OR_hat(or_data, clade, dataset, batch_size = batch_size, random_state = random_state)
    
    BSI = col_to_BSI(SIRsim, or_hat, theta_c = theta_c, theta_bsi = theta_bsi, is_prop = is_prop)
    
    if is_agg:
        BSI = sum_over_bsi(BSI, time_period = time_period)
        #BSI = max_bsi_per_year(BSI, time_period = 52)
        
    # Exponential smoothing. Comment out to remove:
    BSI = exp_smoother(BSI, alpha = alpha)
        
    return BSI


def plot_SIR_and_BSI(y, OR_hat):
    # y: I compartment values of an SIR simulation 
    
    y_bsi = col_to_BSI(y, OR_hat = OR_hat)
    #print(y[0][0]) # S and 1st batch

    #print(y)
    # Plot some simulated colonization and then BSI as determined from that colonization
    plt.plot(y[0][0], label = "Not colonized or colonized by another clade (S)")
    plt.plot(y[1][0], label = "Colonized by clade of interest (I)")
    
    if len(y_bsi) > 1:
        plt.plot(y_bsi[0][0], label = "BSI associated with the clade of interest")
    else:
        plt.plot(y_bsi[0], label = "BSI associated with the clade of interest")
    plt.legend(loc = 'upper right')
    plt.show()
    
## Summaries ##

def BSI_max_t(y):
    # time to peak/maximum number of bsi cases
    
    #return np.argmax(y) # grid
    return np.argmax(y[:,], axis = 1) # ELFI

def BSI_max(y):
    # maximum number of BSI cases
    
    max_bsi = np.max(y[:,], axis = 1) # ELFI
    #max_bsi = np.max(y) # grid
    return max_bsi#.reshape(-1,1).transpose()


def BSI_vector(y):
    # Compare all yearly simulated BSIs to the observed BSI
    # Instead of a scalar summary statistic, we now have a vector of n points (n = number of years in the observed data)
    
    return y

def BSI_cumsum_quantile(y):
    
    return np.quantile(np.cumsum(y[:,]), 0.5)

# 13 summaries, one for each year:

# For ELFI:
def BSI_1(y):
    return y[:,0]

def BSI_2(y):
    return y[:,1]

def BSI_3(y):
    return y[:,2]

def BSI_4(y):
    return y[:,3]

def BSI_5(y):
    return y[:,4]

def BSI_6(y):
    return y[:,5]

def BSI_7(y):
    return y[:,6]

def BSI_8(y):
    return y[:,7]

def BSI_9(y):
    return y[:,8]

def BSI_10(y):
    return y[:,9]

def BSI_11(y):
    return y[:,10]

def BSI_12(y):
    return y[:,11]

def BSI_13(y):
    return y[:,12]

def BSI_14(y):
    return y[:,13]


# For grid:

if grid:

    def BSI_1(y):
        return y[0]

    def BSI_2(y):
        return y[1]

    def BSI_3(y):
        return y[2]

    def BSI_4(y):
        return y[3]

    def BSI_5(y):
        return y[4]

    def BSI_6(y):
        return y[5]

    def BSI_7(y):
        return y[6]

    def BSI_8(y):
        return y[7]

    def BSI_9(y):
        return y[8]

    def BSI_10(y):
        return y[9]

    def BSI_11(y):
        return y[10]

    def BSI_12(y):
        return y[11]

    def BSI_13(y):
        return y[12]

    def BSI_14(y):
        return y[13]


    def BSI_max_t(y):
        # time to peak/maximum number of bsi cases

        return np.argmax(y) # grid
        #return np.argmax(y[:,], axis = 1) # ELFI

    def BSI_max(y):
        # maximum number of BSI cases

        #max_bsi = np.max(y[:,], axis = 1) # ELFI
        max_bsi = np.max(y) # grid
        return max_bsi#.reshape(-1,1).transpose()

