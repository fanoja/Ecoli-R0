# Functions for calculating/visualizing distance between pairs of parameters
import numpy as np
import importlib

import cluster.scripts.BSI_functions
importlib.reload(cluster.scripts.BSI_functions) # for changes to take effect

from cluster.scripts.BSI_functions import *

def distance(sim, obs, S1_fun, S2_fun):
    # Euclidean distance between the simulated and observed sequence
    # d(a*, a) where a* is the observed sequence and a is the simulated sequence
    # sim: simulated sequence
    # obs: observed sequence
    # S1_fun, S2_fun: summary functions that return a scalar summary based on the input sequence
    
    S1_obs = S1_fun(obs)
    S2_obs = S2_fun(obs)
    
    S1_sim = S1_fun(sim)
    S2_sim = S2_fun(sim)
    
    # TODO: add weigthing with standard deviation of S1 and S2 over some amount of simulation
    return np.sqrt((S1_sim - S1_obs)**2 + (S2_sim - S2_obs)**2)


def distance_generalized(y_sim, y_obs, sum_func): # TODO: sum_func as args
    # Allow more than 2 summaries
    # sum_func: summary functions of interest in a list
    # y_sim: simulated sequence
    # y_obs: observed sequence
    
    dist = 0
    
    for i in range(0, len(sum_func)):
        summary = sum_func[i]
        dist += (summary(y_sim) - summary(y_obs))**2

    return np.sqrt(dist)


def get_valid_beta_gamma_pairs(n_beta, n_gamma, min_gamma = 0.01, max_gamma = 0.1, min_R0 = 1, max_R0 = 8):
    # Get pairs of beta and gamma that produce R0 values within [min_R0, max_R0]
    # Returns (n_beta*n_gamma, 2) matrix of (gamma, beta) pairs
    
    gammas = np.linspace(min_gamma, max_gamma, n_gamma)  
    i = 0
    
    par_mat = np.zeros((n_beta*n_gamma, 2))
    
    for g in range(0, len(gammas)):
        gamma = gammas[g]

        potential_betas = gamma*np.linspace(min_R0 + 0.00001, max_R0, n_beta) # all possible values for beta for this given gamma parameter

        for b in range(0, n_beta):
            par_mat[i,0] = potential_betas[b]
            par_mat[i,1] = gamma

            i += 1

    return par_mat

def get_nt_R_pairs(n_nt, n_R, nt_range = [0.01,20], R_range = [1.5,8]):
    # nt = net transmission
    
    pairs = np.zeros((n_nt*n_R, 2))
    
    R = np.linspace(R_range[0], R_range[1], n_R)
    nt = np.linspace(nt_range[0], nt_range[1], n_nt)
    
    count = 0
    for i in range(0, n_nt):

        for j in range(0, n_R):
            
            pairs[count, 0] = nt[i]
            pairs[count, 1] = R[j]
            
            count += 1
            
    return pairs


def get_distance_points(pairs, bsi_obs, sim_pars, summaries):
    # Calculates distances between given pairs of gamma, beta parameters for the summaries of interest
    # pairs: matrix of size (n_gamma*n_beta,2), where the first column holds the gamma values and the 2nd column has the beta values
    # Returns a tuple of (betas, gammas, dists)
    
    dists = np.zeros(pairs.shape[0])
    summary_dists = np.zeros((pairs.shape[0], len(summaries)))
    
    for i in range(0, pairs.shape[0]):
        
        if i%1000 == 0:
                print("Iter:", i)
                
        par1 = pairs[i,0]
        par2 = pairs[i,1]
        
        #print(gamma, beta)
        
        # simulate a sequence
        
        sim_seq = SIR_and_BSI_simulator(par1, par2, sim_pars["n_weeks"], sim_pars["pop_size"], sim_pars["bsi_pars"],\
                                      sim_pars["is_prop"], sim_pars["is_agg"], sim_pars["time_period"], sim_pars["reparam"],\
                                      sim_pars["batch_size"], sim_pars["random_state"])[0]
        
        k = 0
        for summary in summaries:
            summary_dists[i,k] = (summary(bsi_obs) - summary(sim_seq))**2
            k += 1
        

    for k in range(0, len(summaries)):

        #d = (summary(bsi_obs) - summary(sim_seq))**2
        SD = np.std(summary_dists[:,k])

        if SD == 0:
            print("Warning! SD is zero. Summary", summary)
            SD = 1 # TODO: what to do in this case? 

        dists += 1/SD*summary_dists[:,k]

    dists = np.sqrt(dists)
        
        
    return dists, summary_dists
  

## Visualization ##

def scatter_distance_points(betas, gammas, dists, true_beta = None, true_gamma = None, ylab = "Gamma", xlab = "Beta", cutoff_upper = 1, cutoff_lower = 0, save = False, filename = "no_name"):
    
    sc = plt.scatter(betas, gammas, c = dists, s = 1)
    if true_gamma != None and true_beta != None:
        plt.scatter(true_beta, true_gamma, c= "red", marker = "X")
        
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar(sc)
    sc.set_cmap('viridis') # 'plasma'
    #sc.set_clim(cutoff_lower, cutoff_upper)
    if save:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    
def plot_histograms(dists, betas, gammas, eps, par1_label = "Beta", par2_label = "Gamma", xlim = None, save = False, filename = "no_name"):  
    # eps: tolerance. Plot parameter values with distance under this value.
    
    ind = np.where(dists < eps)[0]
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(betas[ind])
    axs[1].hist(gammas[ind])
    axs[0].set_xlabel(par1_label)
    axs[1].set_xlabel(par2_label)
    axs[0].set_title(f"Tolerance: {eps}")
    if xlim != None:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    if save:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    
    

