import numpy as np
import os

def get_OR_hat_pars(or_data, clade = "A", dataset = "NORM"):
    
    df = or_data[or_data["Label"] == f'{clade} ({dataset})']
    or_mu = df["OR"]
    or_sd = (df["upper"] - df["lower"])/2
    
    return or_mu, or_sd
    
def get_OR_hat(or_data, clade = "A", dataset = "NORM", batch_size = 1, random_state = None):
    # TODO: fix that random state
    or_mu, or_sd = get_OR_hat_pars(or_data, clade = clade, dataset = dataset)

    OR_hats = np.empty(batch_size)
    
    for b in range(0, batch_size):
        OR_hat = np.random.normal(or_mu, or_sd**2, 1)

        max_iter = 1000
        i = 0

        while OR_hat[0] < 0:
            OR_hat = np.random.normal(or_mu, or_sd**2, 1)
            i = i + 1
            if i == max_iter:
                break

        if i > 0:
            print(f"Iterated OR_hat {i} times due to negativity.")

        if OR_hat < 0:
            print(f"Warning, negative OR_hat after max iterations!")

        OR_hats[b] = OR_hat
    
    return OR_hats

def col_to_BSI(SIR, OR_hat, theta_c = 1, theta_bsi = 0.3, is_prop = True):
    # SIR: output of the SIR simulator (clade of interest colonization proportion over time)
    # theta_bsi: The proportion of bsi in the entire (colonized) population - from the age distribution.
    # theta_c: The overall proportion of population colonized by E. coli. For simplicity, assume we are only interested in the colonized 
    # population and set theta_c = 1 by default.
    # By changing is_prop = False, can work with counts instead of proportions
    
    theta_c_a = SIR[1]
    
    if not is_prop: 
        N = SIR[0][0][0] + 1 # the first entry in S compartment + 1 is the population size.
        theta_c = N*theta_c 
        theta_bsi = theta_bsi*N
    else:
        if np.min(np.max(SIR[1][:,], axis = 1)) > 1:
            print("Warning! col_to_BSI uses proportions, but SIR seems to use counts.")
    
    bs = theta_c_a.shape[0]
    n_obs = theta_c_a.shape[1] # with batches. 0 = first batch

    theta_c_0 = theta_c - theta_c_a

    theta_bsi_a_hat = OR_hat.reshape(-1,1)*theta_c_a*theta_bsi/(theta_c_0 + OR_hat.reshape(-1,1)*theta_c_a)
    
    return theta_bsi_a_hat


def sum_over_bsi(bsi_obs, time_period = 52):
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

def plot_col_to_BSI(SIR, or_data, clade = "A", dataset = "NORM", n_rep = 100, theta_c = 1, theta_bsi = 0.3, is_prop = True):
    # Plot n_rep repetitions of theta_BSI_clade as "translated" from colonization by clade of interest.
    
    
    all_bsi_reps= []
    for i in range(0, n_rep):
        or_hat = get_OR_hat(or_data, clade = clade, dataset = dataset)
        obsBSI = col_to_BSI(SIR, or_hat, theta_c = theta_c, theta_bsi = theta_bsi, is_prop = is_prop)
        if i == 0:
            plt.plot(obsBSI[0], color = "lightblue", label = "Theta_bsi_A")
        else:
            plt.plot(obsBSI[0], color = "lightblue")
        all_bsi_reps.append(obsBSI)

    plt.plot(SIR[1][0], color = "red", label = "theta_c_A")
    plt.plot(np.mean(all_bsi_reps, axis = 0)[0], color = "navy", label = "Mean of BSI reps")
    plt.xlabel("Years")
    if is_prop:
        plt.title(f"Proportion of BSI: Clade {clade}, {dataset}")
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

def SIR_and_BSI_simulator(par1, par2, nt, N, bsi_pars, is_prop = False, is_agg = False, time_period = 52, reparam = False, batch_size = 1, random_state = None):
    # A simulator function combining both the SIR simulation and the observational model
    
    cwd = os.getcwd()

    if bool(re.search('cluster', cwd)): # a hack to get these loaded from the main directory vs from cluster/
        from scripts.SIR_functions import SIR, prop_to_nSIR
    else:
        from cluster.scripts.SIR_functions import SIR, prop_to_nSIR
    
    # SIR simulator:
    SIRsim = SIR(par1, par2, nt = nt, N = N, reparam = reparam, batch_size = batch_size, random_state = random_state)
    
    if not is_prop:
        SIRsim = prop_to_nSIR(SIR, N)
        
    # Observational model:
    
    or_data = bsi_pars["or_data"]
    clade = bsi_pars["clade"]
    dataset = bsi_pars["dataset"]
    theta_c = bsi_pars["theta_c"]
    theta_bsi = bsi_pars["theta_bsi"]
    
    
    or_hat = get_OR_hat(or_data, clade, dataset, batch_size = batch_size, random_state = random_state)
    
    BSI = col_to_BSI(SIRsim, or_hat, theta_c = theta_c, theta_bsi = theta_bsi, is_prop = is_prop)
    
    if is_agg:
        BSI = sum_over_bsi(BSI, time_period = time_period)

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
    
