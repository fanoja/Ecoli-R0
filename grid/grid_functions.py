# Functions for calculating/visualizing distance between pairs of parameters
import numpy as np
import importlib

import cluster.scripts.BSI_functions
importlib.reload(cluster.scripts.BSI_functions) # for changes to take effect

from cluster.scripts.BSI_functions import *

from cluster.scripts.load_data import * # import data: odds ratios, BSI... # Assuming that data has been loaded!
    

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

def get_uniform_beta_gamma_pairs(n_beta, n_gamma, min_gamma = 0.001, max_gamma = 0.01, min_beta = 0, max_beta = 1):
    # Get uniform priors for beta and gamma
    # returns a (n_beta*n_gamma, 2) matrix of parameter pairs
     
    gammas = np.linspace(min_gamma, max_gamma, n_gamma)
    par_mat = np.zeros((n_beta*n_gamma, 2))
    
    i = 0
    for g in range(0, n_gamma):
        gamma = gammas[g]
        betas = np.linspace(min_beta, max_beta, n_beta)
        for b in range(0, n_beta):
            par_mat[i,0] = betas[b]
            par_mat[i,1] = gamma
            
            i += 1
    
    return par_mat
            
    
def get_valid_beta_gamma_pairs_old(n_beta, n_gamma, min_gamma = 0.001, max_gamma = 0.1, min_R0 = 1.01, max_R0 = 20):
    # DEPRECATED, see the funciton below.
    # Get pairs of beta and gamma that produce R0 values within [min_R0, max_R0]
    # Returns (n_beta*n_gamma, 2) matrix of (gamma, beta) pairs
    # R = [1.01, 20] From Lintusaari et al 2019
    
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

def get_valid_beta_gamma_pairs(n_grid, min_R0 = 1.01, max_R0 = 5, min_gamma = 0.0001, max_gamma = 0.1):
    # Get beta and gamma pairs such that the R0 produced by these parameter pairs are between min_R0 and max_R0.
    # Note: order of beta and gamma matters! Each pair produces a reasonable R0, but I can't guarantee that after reordering beta and gamma this is still the case.
    # Note: the maximum of max_R0 is limited to 5, since earlier runs of this model indicate a relatively small beta parameter.
    
    R0_prior = np.random.uniform(min_R0, max_R0, n_grid)
    gammas = np.random.uniform(min_gamma, max_gamma, n_grid)
    betas = np.zeros(n_grid)

    for i in range(0, n_grid):
        betas[i] = R0_prior[i]*gammas[i]

    par_mat = np.zeros((n_grid, 2))
    
    for i in range(0, n_grid):
        par_mat[i,0] = betas[i]
        par_mat[i,1] = gammas[i]
    
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

#par1, par2, nt, N, bsi_pars, I0 = None, is_prop = False, is_agg = False, time_period = 52, reparam = False, batch_size = 1, random_state = None
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
        
        sim_seq = SIR_and_BSI_simulator(par1 = par1, par2 = par2, nt = sim_pars["n_weeks"], N = sim_pars["pop_size"], bsi_pars = sim_pars["bsi_pars"],\
                                        is_prop = sim_pars["is_prop"], is_agg = sim_pars["is_agg"], time_period = sim_pars["time_period"],\
                                        reparam = sim_pars["reparam"], batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])[0]
        
        k = 0
        for summary in summaries:
            summary_dists[i,k] = np.sum((summary(bsi_obs) - summary(sim_seq))**2)
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

def read_sim_pars(filepath):
    # Reads simulation parameters from a text file into a dictionary. 
    # Why: for visualization and rerunning a specific simulation
    import ast
    
    print("Results to visualize: " + os.path.join(filepath, "sim_params.txt"))
    with open(os.path.join(filepath, "sim_params.txt"), "r") as f:
        lines = f.readlines()
        sim_pars = {}
        for line in lines:
            parts = line.strip().split(': ')
            try:
                if parts[0] in ["theta_c", "theta_bsi"]:
                    sim_pars[parts[0]] = float(parts[1])
                else:
                    sim_pars[parts[0]] = int(parts[1])
            except ValueError:
                if parts[1] in ["True", "False", "None"]:
                    sim_pars[parts[0]] = ast.literal_eval(parts[1])
                else:
                    sim_pars[parts[0]] = parts[1]
        
    return sim_pars

def scatter_distance_points(betas, gammas, dists, true_beta = None, true_gamma = None, ylab = "Gamma", xlab = "Beta", cutoff_upper = 1, cutoff_lower = 0, save = False, filename = "no_name", title = None):
    
    sc = plt.scatter(betas, gammas, c = dists, s = 1)
    if true_gamma != None and true_beta != None:
        plt.scatter(true_beta, true_gamma, c= "red", marker = "X")
        
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar(sc)
    sc.set_cmap('viridis') # 'plasma'
    sc.set_clim(cutoff_lower, cutoff_upper)
    if title != None:
        plt.title(title)
    if save:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    
def plot_histograms(dists, betas, gammas, eps, par1_label = "Beta", par2_label = "Gamma", xlim = None, save = False, filename = "no_name"):  
    # eps: tolerance. Plot parameter values with distance under this value.
    
    ind = np.where(dists < eps)[0]
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(betas[ind])
    axs[1].hist(gammas[ind])
    #axs[1,0].hist(betas[ind]/gammas[ind])
    axs[0].set_xlabel(par1_label)
    axs[1].set_xlabel(par2_label)
    #axs[1,0].set_xlabel("R0")
    axs[0].set_title(f"Tolerance: {eps}")
    if xlim != None:
        axs[0,0].set_xlim(xlim)
        axs[0,1].set_xlim(xlim)
    if save:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    
def plot_posterior_scatterplot(summary_dists, pairs, dists, eps, output_directory):
    # Plot a scatterplot of the posterior parameters and their joint distributions
    
    acc_pairs = pairs[np.where(dists< eps)[0],:]
    
    fig, axs = plt.subplots(2, 2)
    axs[0,0].hist(acc_pairs[:,0])
    axs[0,0].set_ylabel("Beta")
    axs[0,1].scatter(acc_pairs[:,0], acc_pairs[:,1])
    #axs[0,1].set_ylabel("Beta")
    #axs[0,1].set_xlabel("Gamma")
    axs[1,0].scatter(acc_pairs[:,1], acc_pairs[:,0])
    axs[1,0].set_ylabel("Gamma")
    axs[1,0].set_xlabel("Beta")
    axs[1,1].hist(acc_pairs[:,1])
    axs[1,1].set_xlabel("Gamma")
    axs[0,0].set_title(f"Posterior, tolerance: {eps}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "posterior_plots.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    
# Summary distance visualizations


def plot_summary_dists(summary_dists, output_directory, scale = False, S1_name = "Max BSI", S2_name = "Max t", filename = "summary_dists_plot.pdf"):
    # Plot histograms and scatterplots of summary dists
    # Assumes two summaries
    
    if scale:
        summary_dists[:,0] = 1/np.std(summary_dists[:,0])*summary_dists[:,0]
        summary_dists[:,1] = 1/np.std(summary_dists[:,1])*summary_dists[:,1]
        
    fig, axs = plt.subplots(2, 2)
    axs[0,0].hist(summary_dists[:,0])
    axs[0,0].set_ylabel(S1_name)
    axs[0,1].scatter(summary_dists[:,0], summary_dists[:,1])
    #axs[0,1].set_ylabel("Beta")
    #axs[0,1].set_xlabel("Gamma")
    axs[1,0].scatter(summary_dists[:,1], summary_dists[:,0])
    axs[1,0].set_ylabel(S2_name)
    axs[1,0].set_xlabel(S1_name)
    axs[1,1].hist(summary_dists[:,1])
    axs[1,1].set_xlabel(S2_name)
    if scale:
        axs[0,0].set_title(f"Summary dists, scaled")
    else:
        axs[0,0].set_title(f"Summary dists, unscaled")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, filename), format="pdf", bbox_inches="tight")
    plt.show()
    
# Functions for visualizing observed and simulated data:
def get_bounds(par_posterior, ci):
    
    lower_ci = (100-ci)/2
    upper_ci = 100 - (100-ci)/2
    
    lb = np.percentile(par_posterior, lower_ci)
    ub = np.percentile(par_posterior, upper_ci)
    
    return lb, ub

def plot_observed_and_simulated_seq(bsi_obs_data, dists, pairs, eps, sim_pars, output_directory, ci = 90):
    # Simulate based on the beta and gamma parameters - simulate more than one potential colonization to quantify uncertainty:

    # The same indices, however, draws from a uniform distribution over the potential BSI curves (for example, from min(beta) to max(beta)),
    # does not take into account the normal distribution of the parameters
    acc_pairs = pairs[np.where(dists< eps)[0],:]
    n_reps = 100
    indx = np.random.choice(np.arange(1,len(acc_pairs[:,0])), size = n_reps) # pairs to choose from

    par1s = acc_pairs[indx, 0]
    par2s = acc_pairs[indx, 1]
    
    clade = sim_pars["clade"]
    dataset = sim_pars["dataset"]
    #output_directory = sim_pars["output_directory"]
    
    bsi_pars = {"or_data":or_data, "clade": sim_pars["clade"], "dataset":sim_pars["dataset"], "theta_c":sim_pars["theta_c"], "theta_bsi":sim_pars["theta_bsi"], "include_I0": sim_pars["include_I0"]}
    
    
    mean_simseq = SIR_and_BSI_simulator(par1 = np.mean(pairs[np.where(dists< eps)[0],0]),\
                                        par2 = np.mean(pairs[np.where(dists < eps)[0],1]),\
                                        nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                        bsi_pars = bsi_pars,\
                                        is_prop = sim_pars["is_prop"],\
                                        is_agg = sim_pars["is_agg"],\
                                        time_period = sim_pars["time_period"],\
                                        reparam = sim_pars["reparam"],\
                                        batch_size = sim_pars["batch_size"],\
                                        random_state = sim_pars["random_state"])
    
    median_simseq = SIR_and_BSI_simulator(par1 = np.median(pairs[np.where(dists< eps)[0],0]),\
                                          par2 = np.median(pairs[np.where(dists < eps)[0],1]),\
                                          nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                          bsi_pars = bsi_pars, is_prop = sim_pars["is_prop"],\
                                          is_agg = sim_pars["is_agg"],\
                                          time_period = sim_pars["time_period"],\
                                          reparam = sim_pars["reparam"], batch_size = sim_pars["batch_size"],\
                                          random_state = sim_pars["random_state"])

    #for i in range(0,n_reps):
        #simseq = SIR_and_BSI_simulator(par1 = par1s[i], par2 = par2s[i],\
                                       #nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                       #bsi_pars = bsi_pars, is_prop = sim_pars["is_prop"],\
                                       #is_agg = sim_pars["is_agg"],\
                                       #time_period = sim_pars["time_period"],\
                                       #reparam = sim_pars["reparam"], batch_size = sim_pars["batch_size"],\
                                       #random_state = sim_pars["random_state"])

        #plt.plot(simseq[0], color = "lightblue")
        
    # OR is fixed to or_mu, or_upper or or_lower to visualize uncertainty associated with the odds ratio
    # deterministic with fixed or_hat values:
    
    df = or_data[or_data["Label"] == f'{clade} (BSAC)'] #{dataset}
    or_mu = df["OR"].values
    
    print(f"Or mu in plot obs sim: {or_mu}")

    or_lower = df["lower"].values
    or_upper = df["upper"].values
    
    print(f"OR lower: {or_lower}, OR upper: {or_upper}")

    mu_sim = SIR_and_BSI_simulator(np.mean(pairs[np.where(dists< eps)[0],0]), np.mean(pairs[np.where(dists < eps)[0],1]),\
                                            nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                            bsi_pars = bsi_pars,\
                                            is_prop = sim_pars["is_prop"],\
                                            is_agg = sim_pars["is_agg"],\
                                            time_period = sim_pars["time_period"],\
                                            reparam = sim_pars["reparam"],\
                                            has_or_hat = True, manual_or_hat = or_mu,\
                                            batch_size = sim_pars["batch_size"],\
                                            random_state = sim_pars["random_state"])
    
    print(mu_sim)

    median_sim = SIR_and_BSI_simulator(np.median(acc_pairs[:,0]), np.median(acc_pairs[:,1]),\
                                            nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                            bsi_pars = bsi_pars,\
                                            is_prop = sim_pars["is_prop"],\
                                            is_agg = sim_pars["is_agg"],\
                                            time_period = sim_pars["time_period"],\
                                            reparam = sim_pars["reparam"],\
                                            has_or_hat = True, manual_or_hat = or_mu,\
                                            batch_size = sim_pars["batch_size"],\
                                            random_state = sim_pars["random_state"])                                       

    lower_sim = SIR_and_BSI_simulator(np.mean(pairs[np.where(dists< eps)[0],0]), np.mean(pairs[np.where(dists < eps)[0],1]),\
                                            nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                            bsi_pars = bsi_pars,\
                                            is_prop = sim_pars["is_prop"],\
                                            is_agg = sim_pars["is_agg"],\
                                            time_period = sim_pars["time_period"],\
                                            reparam = sim_pars["reparam"],\
                                            has_or_hat = True, manual_or_hat = or_lower,\
                                            batch_size = sim_pars["batch_size"],\
                                            random_state = sim_pars["random_state"])

    upper_sim = SIR_and_BSI_simulator(np.mean(pairs[np.where(dists< eps)[0],0]), np.mean(pairs[np.where(dists < eps)[0],1]),\
                                            nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                            bsi_pars = bsi_pars,\
                                            is_prop = sim_pars["is_prop"],\
                                            is_agg = sim_pars["is_agg"],\
                                            time_period = sim_pars["time_period"],\
                                            reparam = sim_pars["reparam"],\
                                            has_or_hat = True, manual_or_hat = or_upper,\
                                            batch_size = sim_pars["batch_size"],\
                                            random_state = sim_pars["random_state"])
   
    print(mu_sim[0,:])
    plt.plot(lower_sim[0], color = "lightblue", label = "OR CIs, upper and lower")
    plt.plot(upper_sim[0], color = "lightblue")
    plt.plot(mu_sim[0], color = "blue")
    plt.plot(median_sim[0], color = "pink")
        
    # Plot the upper and lower bounds for both parameters:
    par1_posterior = acc_pairs[:,0]
    par2_posterior = acc_pairs[:,1]
    
    par1_bounds = get_bounds(par1_posterior, ci)
    par2_bounds = get_bounds(par2_posterior, ci)
    
    # NOTE: Or_hat is set to or_mu to get the same uncertainty (otherwise, or_hat is sampled again at each visualization -> visualizations are different every time)
    # NOTE: if the dashed lines look weird, it is because to boundary parameters generate a weird SIR curve.
    lb_simseq = SIR_and_BSI_simulator(par1 = par1_bounds[0], par2 = par2_bounds[0],\
                                   nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                   bsi_pars = bsi_pars, is_prop = sim_pars["is_prop"],\
                                   is_agg = sim_pars["is_agg"],\
                                   time_period = sim_pars["time_period"],\
                                   reparam = sim_pars["reparam"],\
                                      has_or_hat = True, manual_or_hat = or_mu,\
                                    batch_size = sim_pars["batch_size"],\
                                   random_state = sim_pars["random_state"])
    ub_simseq = SIR_and_BSI_simulator(par1 = par1_bounds[1], par2 = par2_bounds[1],\
                                nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],\
                                bsi_pars = bsi_pars, is_prop = sim_pars["is_prop"],\
                                is_agg = sim_pars["is_agg"],\
                                time_period = sim_pars["time_period"],\
                                reparam = sim_pars["reparam"],\
                                has_or_hat = True, manual_or_hat = or_mu,\
                                batch_size = sim_pars["batch_size"],\
                                random_state = sim_pars["random_state"])

    plt.plot(lb_simseq[0], color = "grey", linestyle='--')
    plt.plot(ub_simseq[0], color = "grey", linestyle='--', label = f"Posterior CIs ({ci}%)")
        
    #plt.plot(mean_simseq[0], label = "Simulated mean BSI", color = "blue")
    #plt.plot(median_simseq[0], label = "Simulated median BSI", color = "violet")
    plt.plot(np.array(bsi_obs_data), label = "True BSI", color = "orange")
    plt.legend()
    plt.title(f"Simulated and observed BSI, clade {clade}")
    plt.savefig(os.path.join(output_directory, "sim_and_obs_BSI.pdf"), format="pdf", bbox_inches="tight")
    plt.xlabel("Year")
    plt.ylabel("Proportion of population")
    plt.show()
    
    
def prior_histograms(pairs, output_directory):

    # R0 histogram
    R0 = pairs[:,0]/pairs[:,1]
    print(R0.shape)
    plt.hist(R0)
    plt.title("R0 = beta/gamma prior")
    plt.savefig(os.path.join(output_directory, "R0_prior.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    
    # beta/gamma histogram
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(pairs[:,0])
    axs[1].hist(pairs[:,1])
    #axs[2].hist(pairs[:,0]/pairs[:,1])
    axs[0].set_title(f"Par1 (beta) prior")
    axs[1].set_title(f"Par2 (gamma) prior")
    #axs[2].set_title(f"R0 prior")
    plt.savefig(os.path.join(output_directory, "param_priors.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    
def plot_colonization(bsi_obs_data, dists, pairs, eps, sim_pars, output_directory, ci = 90):

    from cluster.scripts.SIR_functions import SIR, prop_to_nSIR

    if sim_pars["include_I0"]:
        theta_bsi_a_0 = bsi_obs_data.iloc[0]/sim_pars["time_period"]
        or_hat = get_OR_hat(or_data = or_data, clade = sim_pars["clade"], dataset = "BSAC", batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])
        I0 = (theta_bsi_a_0*sim_pars["theta_c"]/(theta_bsi_a_0 + or_hat[0]*sim_pars["theta_bsi"] - theta_bsi_a_0*or_hat[0]))*sim_pars["pop_size"]
    else:
        I0 = None


    n_draws = 100 # Number of beta-gamma pairs to draw from the posterior
    acc_pairs = pairs[np.where(dists< eps)[0],:] # accepted parameter pairs
    indx = np.random.choice(np.arange(1,len(acc_pairs[:,0])), size = n_draws) # pairs to choose from; only use the accepted parameter pairs
    par1s = acc_pairs[indx, 0]
    par2s = acc_pairs[indx, 1]

    #for i in range(0, n_draws):

        #colseq = SIR(par1 = par1s[i], par2 = par2s[i], I0 = I0,\
                 #nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],reparam = sim_pars["reparam"],\
                 #batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])

        #plt.plot(colseq[1][0], color = "pink")
        
    par1_posterior = acc_pairs[:,0]
    par2_posterior = acc_pairs[:,1]
    
    par1_bounds = get_bounds(par1_posterior, ci)
    par2_bounds = get_bounds(par2_posterior, ci)
    
    lb_simseq = SIR(par1 = par1_bounds[0], par2 = par2_bounds[0], I0 = I0,\
                 nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],reparam = sim_pars["reparam"],\
                    is_prop = sim_pars["is_prop"],\
                 batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])
    ub_simseq = SIR(par1 = par1_bounds[1], par2 = par2_bounds[1], I0 = I0,\
                 nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],reparam = sim_pars["reparam"],\
                    is_prop = sim_pars["is_prop"],\
                 batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])

    plt.plot(lb_simseq[1][0], color = "pink", linestyle='--', label = f"{ci}% posterior CIs")
    plt.plot(ub_simseq[1][0], color = "pink", linestyle='--')


    colseq_mean = SIR(par1 = np.mean(pairs[np.where(dists< eps)[0],0]), par2 = np.mean(pairs[np.where(dists< eps)[0],1]), I0 = I0,\
                 nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],reparam = sim_pars["reparam"],\
                      is_prop = sim_pars["is_prop"],\
                 batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])
    colseq_median = SIR(par1 = np.median(pairs[np.where(dists< eps)[0],0]), par2 = np.median(pairs[np.where(dists< eps)[0],1]), I0 = I0,\
                 nt = sim_pars["n_weeks"], N = sim_pars["pop_size"],reparam = sim_pars["reparam"],\
                        is_prop = sim_pars["is_prop"],\
                 batch_size = sim_pars["batch_size"], random_state = sim_pars["random_state"])

    clade = sim_pars["clade"]
    
    plt.plot(colseq_mean[1][0], label = f"Mean colonization", color = "red")
    plt.plot(colseq_median[1][0], label = f"Median colonization", color = "violet")
    plt.title(f"Colonization by clade {clade}")
    plt.xlabel("Week")
    plt.ylabel("Proportion of population")
    plt.legend()
    plt.savefig(os.path.join(output_directory, "colonization.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    
def visualize_results(output_directory, eps):
    # Create all relevevant visualizations and save to output_directory
    # Give a directory to visualize results from and a tolerance value.
    # Save resulting figures in this directory.
    
    # Read the specific simulation parameters from file into a dictionary:
    

    sim_pars = read_sim_pars(output_directory)
    print(sim_pars)
    clade = sim_pars["clade"]
    
    if sim_pars["dataset"] == "NORM":
        bsi_obs_data = get_incidence_data("data/NORM_incidence.csv", clade = sim_pars["clade"],\
                                      is_prop = sim_pars["is_prop"],\
                                      n_incidence_pop = sim_pars["pop_size"])
    else:
        bsi_obs_data = get_obs_BSI(df = bsac_data, clade = sim_pars["clade"], is_prop = sim_pars["is_prop"])

    print(bsi_obs_data)
    
    # Load distance matrix and the parameter pairs:
    
    dists = np.load(os.path.join(output_directory, "dists.npy"))
    pairs = np.load(os.path.join(output_directory, "pairs.npy"))
    summary_dists = np.load(os.path.join(output_directory, "summary_dists.npy"))

    
    ## Distance histogram
    
    plt.hist(dists)
    plt.title("Distance")
    plt.savefig(os.path.join(output_directory, "dist_histogram.pdf"))
    plt.show()
    
    ## Parameter histograms
                
    print(f"Beta mean: {np.mean(pairs[np.where(dists< eps)[0],0])}")
    print(f"Gamma mean: {np.mean(pairs[np.where(dists < eps)[0],1])}")
    print(f"R = mean(beta/gamma): {np.mean(pairs[np.where(dists < eps)[0],0]/pairs[np.where(dists < eps)[0],1])}")
    print(f"R = median(beta/gamma): {np.median(pairs[np.where(dists < eps)[0],0]/pairs[np.where(dists < eps)[0],1])}")
    #print(f"True parameters (for synthetic data): beta = {true_par1}, gamma = {true_par2}")
                
    plot_histograms(dists, pairs[:,0], pairs[:,1], eps, save = True,\
                    filename = os.path.join(output_directory, "param_histograms.pdf"))
    
    
    # Plot a histogram of R:
    
    ind = np.where(dists < eps)[0]
    plt.hist(pairs[np.where(dists < eps)[0],0]/pairs[np.where(dists < eps)[0],1])
    plt.title(f"R0, tolerance: {eps}")
    plt.xlabel("R0")
    plt.savefig(os.path.join(output_directory, "R0_hist.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    
    
    ## Posterior distribution (distribution of accepted parameters as a joint distribution in addition to the histograms)
    
    plot_posterior_scatterplot(summary_dists, pairs, dists, eps, output_directory)
    
    ## Observed and simulated data
                    
    
    plot_observed_and_simulated_seq(bsi_obs_data, dists, pairs, eps, sim_pars, output_directory)
                    

    ## Colonization
                    
    plot_colonization(bsi_obs_data, dists, pairs, eps, sim_pars, output_directory)
    
    ## Scatterplot of parameter pairs
    
    scatter_distance_points(pairs[:,0], pairs[:,1], dists, true_beta = None, true_gamma = None,\
                        save = True, filename = os.path.join(output_directory, "grid_scatter.pdf"),\
                            cutoff_upper = eps, cutoff_lower = 0)
    
    ## Grid scatterplots of summary distances, scaled and unscaled

    # Max BSI, unscaled:
    #scatter_distance_points(pairs[:,0], pairs[:,1], summary_dists[:,0], true_beta = None, true_gamma = None, ylab = "Gamma", xlab = "Beta", cutoff_upper = eps, cutoff_lower = 0, save = True, filename = os.path.join(output_directory, "grid_scatter_S1_unscaled.pdf"), title = "S1 unscaled")

    
    #if summary_dists.shape[1] == 2:
        # Max t, unscaled:
        #scatter_distance_points(pairs[:,0], pairs[:,1], summary_dists[:,1], true_beta = None, true_gamma = None, ylab = "Gamma", xlab = "Beta", cutoff_upper = eps, cutoff_lower = 0, save = True, filename = os.path.join(output_directory, "grid_scatter_S2_unscaled.pdf"), title = "S2, unscaled")

    # Scaled scatterplot:
    # TODO: generalize this to work with any number of summary statistics
    plot_summary_scatter = False
    if plot_summary_scatter:
        for s in range(0, summary_dists.shape[1]):

            scatter_distance_points(pairs[:,0], pairs[:,1], 1/np.std(summary_dists[:,s])*summary_dists[:,s], true_beta = None, true_gamma = None, ylab = "Gamma", xlab = "Beta", cutoff_upper = eps, cutoff_lower = 0, save = True, filename = os.path.join(output_directory, f"grid_scatter_S{s + 1}_scaled.pdf"), title =  f"S{s + 1}, scaled")

    #if summary_dists.shape[1] == 2:
        #scatter_distance_points(pairs[:,0], pairs[:,1], 1/np.std(summary_dists[:,1])*summary_dists[:,1], true_beta = None, true_gamma = None, ylab = "Gamma", xlab = "Beta", cutoff_upper = eps, cutoff_lower = 0, save = True, filename = os.path.join(output_directory, "grid_scatter_S2_scaled.pdf"), title = "S2, scaled")

    ## Visualize summary distances:
    
    # All simulations
    #plot_summary_dists(summary_dists, output_directory, scale = False, filename = "all_sim_summary_dists_plot_unscaled.pdf")
    #plot_summary_dists(summary_dists, output_directory, scale = True, filename = "all_sim_summary_dists_plot_scaled.pdf") # Note top right y-axis!
    
    # Accepted simulations
    #indx = np.where(dists< eps)[0]
    #acc_summary_dists = summary_dists[indx,:]
    #plot_summary_dists(acc_summary_dists, output_directory, scale = False, filename = "acc_sim_summary_dists_plot_unscaled.pdf")
    #plot_summary_dists(acc_summary_dists, output_directory, scale = True, filename = "acc_sim_summary_dists_plot_scaled.pdf")
    
    
    
    ## Prior histograms:
    
    prior_histograms(pairs, output_directory)

    
### Utility functions ###

# Function for quickly modifying grid_params files in a specific directory

def modify_grid_params(directory, var_to_change, new_value):
    # Change the value of a variable in all files in directory
    
    # directory: directory the files of which are modified
    # var_to_change: variable you want to change, such as theta_bsi
    # new_value: new value assigned to var_to_change
    

    # Open each file in the directory:
    files = os.listdir(directory)

    for file in files:
        file = os.path.join(directory, file)
        new_content = ""
        try:
            with open(file, "r") as f: # read content and modify
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(' = ')
                    var = parts[0]
                    value = parts[1]
                    if parts[0] != var_to_change:
                        new_content += f"{var} = {value}\n"
                    elif str(parts[0]) == var_to_change:
                        new_content += f"{var} = {new_value}\n"

        except IsADirectoryError:
            print(f"DirectoryError: {file}")

        # Write new content to file:
        try:
            with open(file, "w") as f:
                f.write(new_content)
        except IsADirectoryError:
            print(f"DirectoryError: {file}")

