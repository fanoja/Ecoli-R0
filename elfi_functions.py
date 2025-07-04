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
import os
from datetime import datetime
import time
from collections import OrderedDict

# Load functions for the observation model
import BSI_functions
importlib.reload(BSI_functions) # for changes to take effect
from BSI_functions import get_OR_hat_pars, col_to_BSI, sync_timewindow, sum_over_bsi

# Load functions for the SIR model
import SIR_functions
importlib.reload(SIR_functions) # for changes to take effect
from SIR_functions import SIR, SIS

# Visualisation functions
import vis
importlib.reload(vis)
from vis import plot_priors_elfi, plot_prior_pred, plot_marginals, plot_discrepancy, plot_post_col_w_years, plot_post_col_w_CIs, visualize_ppc, plot_prior_sensitivity, plot_convergence_of_threshold_pops, plot_diagnostics, plot_4x4_summaries


# Locally optimal ABC-SMC
from abc_smc import local_optimal_ABCSMC

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

def get_incidence_data(csv_file, n_years = 14, clade = "A", is_prop = True, n_incidence_pop = 1000000):
    """Get the BSI clade X incidence per n_incidence_pop people. If is_prop = True, divides the incidence by n_incidence_pop.

    Args:
        csv_file (string): csv result file path.
        clade (string): clade of interest, A or C2.
        is_prop (bool): if True, get incidence as proportion of population.
        n_incidence_pop (int): the population size.
        partial_time (bool): simulate only a part of the observed years.
        remove_C2_first (bool): if True, remove first three years for clade C2.

    Returns:
        df (DataFrame): observed BSI data as a data frame.

    """

    #csv_file = 'data/NORM_incidence.csv'
    df = pd.read_csv(csv_file, delimiter=',')
    
    rnames = df["Year"]
    
    df = df[clade]

    if is_prop:
        df = df/n_incidence_pop
     
    df.index = rnames

    if clade == "B":
        df = df.iloc[6:14,]
    if clade == "C2" and n_years == 11:
        df = df.iloc[3:14,]
    
    return df


def get_model_parameters(file):
    """Read the model parameters from <model_config>.txt file to a dictionary.

    Args:
        file (str): model configuration file name, for example "model_config_SIR_A.txt"

    Returns:
        model_config (dict): model parameters in a dictionary.

    """
    # Read the simulation parameter file

    model_config = {}
    
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split('=', 1)
                model_config[key.strip()] = value.strip()

    int_keys = ["pop_size", "n_years", "time_period", "n_sims", "batch_size", "seed"]
    bool_keys = ["is_prop", "is_agg", "reparam"]
    float_keys = ["theta_bsi", "theta_c", "Dt_value", "I0"]

    for k in int_keys:
        model_config[k] = int(model_config[k])
    for k in float_keys:
        model_config[k] = float(model_config[k])
    for k in bool_keys:
        model_config[k] = eval(model_config[k])

    model_config["schedule"] = eval(model_config["schedule"])
    model_config["variables"] = eval(model_config["variables"])

    return model_config

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

    
def unscale_Dt(Dt_scaled, factor):
    # Helper function for scaling the delay parameter Dt, used by get_model()
    
    return Dt_scaled*factor

def get_model(model_config_file, random_state = None):
    """ Get ELFI model for the given data and model configuration files.

    Args:
        model_config_file (str): model configuration parametes in a text file

    Returns:
        m (elfi.Model): ELFI model, built based on the model configuration.

    """

    # Create model configuration dictionary
    model_config = get_model_parameters(model_config_file)
    
    if model_config["Dt_value"] == 0:
        no_Dt = True
    else:
        no_Dt = False
    
    # Load the data files
    or_data = pd.read_excel(model_config["or_data_file"])
    bsi_obs = np.array([get_incidence_data(model_config["bsi_data_file"], n_years = model_config['n_years'], clade = model_config['clade'], is_prop = model_config['is_prop'], n_incidence_pop = model_config['pop_size'])])

    print(bsi_obs)
    
    mu_OR, sd_OR, lower_OR, upper_OR = get_OR_hat_pars(or_data, clade = model_config["clade"])

    # Initialize the model
    m = elfi.new_model()
    elfi.set_client('native')

    # Set priors for par1 (net transmission rate) and par2 (basic reproduction number)
    par1 = elfi.Prior(scipy.stats.uniform, 0, 0.1, model = m)
    loc = 0
    scale = 1
    a_trunc = 1.01
    b_trunc = 3.00
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    par2 = elfi.Prior(scipy.stats.truncnorm, a, b, model = m)


    # Set the delay parameter

        
    if not no_Dt:
        if model_config["clade"] == "B":
            Additional_years = 6.5
        else:
            Additional_years = model_config["Dt_value"]#0.5 # delay: 0.5 years
        Dt = elfi.Prior(scipy.stats.uniform, 0, 0.001*Additional_years * 52, model = m) # fixed a bug 02052025: 0.0001 -> 0.001
        Dt_unscaled = elfi.Operation(unscale_Dt, 1000, Dt, model = m)
        n_sim_years = elfi.Constant(model_config["n_years"], model = m)
        t_array = np.arange((model_config["n_years"] +  Additional_years) * 52)
    else:
        Additional_years = 0
        t_array = np.arange((model_config["n_years"]) * 52)

    print(f"Additional years: {Additional_years}")
    print(f"Dt_value: {model_config['Dt_value']}")

    # Set constant parameters: I0, N, theta_c, time_period and theta_bsi
    I0 = elfi.Constant(model_config["I0"], model = m)
    N = elfi.Constant(model_config["pop_size"], model = m)
    theta_c = elfi.Constant(model_config["theta_c"], model = m)
    theta_bsi = elfi.Constant(model_config["theta_bsi"], model = m)
    time_period = elfi.Constant(model_config["time_period"], model = m)

    # Set the OR
    if model_config["OR_type"] == "lower": # lower 95CI OR
        print("Using lower OR.")
        mu_OR = elfi.Constant(lower_OR)
    elif model_config["OR_type"] == "upper": # upper 95CI OR
        print("Using upper OR.")
        mu_OR = elfi.Constant(upper_OR)
    else: # use mean OR
        print("Using mean OR.")
        mu_OR = elfi.Constant(mu_OR)

    # Set the colonisation model and unobserved weekly BSI
    if model_config["simulator_model"] == "SIR":
        SIRsim = elfi.Operation(SIR, par1, par2, I0, t_array, N, model_config["reparam"], model = m)
        BSI = elfi.Operation(col_to_BSI, SIRsim, mu_OR, theta_c, theta_bsi, np.array(model_config["pop_size"]), model = m)
    elif model_config["simulator_model"] == "SIS": 
        SIRsim = elfi.Operation(SIS, par1, par2, I0, t_array, N, model_config["reparam"], model = m) # name is SIRsim to avoid redundant coding.
        BSI = elfi.Operation(col_to_BSI, SIRsim, mu_OR, theta_c, theta_bsi, np.array(model_config["pop_size"]), model = m)
    else:
        print(f"{model_config['simulator_model']} is not a valid simulator model!")

    # Set yearly BSI 
    if not no_Dt: # include Dt
        synced_data = elfi.Operation(sync_timewindow, BSI, Dt_unscaled, n_sim_years, model = m) # no conv, Dt
        yearly_BSI = elfi.Simulator(sum_over_bsi, synced_data, time_period, model = m, observed = bsi_obs)
    else: # do not include Dt
        yearly_BSI = elfi.Simulator(sum_over_bsi, BSI, time_period, model = m, observed = bsi_obs) # no conv, no Dt

    # Set summaries & discrepancy
    log_BSI = elfi.Summary(custom_log, yearly_BSI, model = m)
    log_BSI_max = elfi.Summary(BSI_max, log_BSI, model = m)
    log_BSI_t0 = elfi.Summary(BSI_t0, log_BSI, model = m)
    BSI_max_t_summary = elfi.Summary(BSI_max_t, yearly_BSI, model = m)
    
    d = elfi.Distance("euclidean", log_BSI_max, BSI_max_t_summary, log_BSI_t0, model = m)

    # Draw and return the model
    print("Model imported")
    elfi.draw(m)

    return m


def run_model(model, model_config_file, save_model = True):
    """ Main function for running the model. Saves the results, the ELFI model, prior & posterior and other relevant outputs and creates preliminary visualizations of the results and a set of diagnostic plots.
    
    Args:
        model (elfi.Model): ELFI model from get_model.
        model_config_file (str): .txt file containig the model configuration parameters.
        save_model (bool): if True, saves the results and preliminary visualizations.
        
    """

    # Load model configuration parameters
    model_config = get_model_parameters(model_config_file)
    
    # Print basic information (clade, number of samples)
    print(f"Number of samples: {model_config['n_sims']}")
    print(f"Clade {model_config['clade']}")
    if model_config["clade"] not in ["A", "C2", "B", "B_349", "B_non_349", "C1"]:
        print("Invalid clade. Choosing clade A.")
        model_config["clade"] = "A"

    if save_model:
        # Create a result file & save model config parameters
        output_directory = create_result_directory(model_config, model_config_file)
        
        # Generate a prior sample from the model and save it for diagnostic purposes
        save_priors(model, model_config, output_directory)
    
    # Run the model
    smc = elfi.SMC(model['d'], batch_size = model_config["batch_size"], seed = model_config["seed"])
    #print("Using locally optimal ABC-SMC")
    #smc = local_optimal_ABCSMC(model, N=model_config["n_sims"], thresholds=model_config["schedule"], discrepancy_name='d', plot_states=False, verbose=False)
    
    start_time = time.time()
    
    result = smc.sample(model_config["n_sims"], thresholds = model_config["schedule"])
    
    end_time = time.time()
    
    # Calculate and print the elapsed time
    elapsed_time = (end_time - start_time)/60
    print("Elapsed time of SMC-ABC (truncated):", np.trunc(elapsed_time), "minutes")

    if save_model:
        # Save the ELFI model and results as .npy files in output_directory
        save_results(result, model, model_config, output_directory)
        
        # Preliminary visualisations
        plot_elfi(output_directory, model_config)

    return result





### Utility functions for saving priors, creating output directory etc ###

def create_result_directory(model_config, model_config_file):
    """ Create a result directory and save the model config file. Save the model configuration to the result directory.

    Args:
        model_config (dict): model configuration as a dictionary.
        model_config_file (str): name of the model configuration file.

    Returns:
        output_directory (str): directory where the model output and visualizations will be saved.s

    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_config["simulator_model"] == "SIS":
        print("Using SIS for colonization model.")
        output_directory = f"res/elfi_res/{model_config['clade']}_results/SIS/clade_{model_config['clade']}_{timestamp}/"
    else:
        output_directory = f"res/elfi_res/{model_config['clade']}_results/clade_{model_config['clade']}_{timestamp}/"

    try:
        os.makedirs(output_directory)
        print(f"Created a result directory: {output_directory}")
    except FileExistsError:
        print(f"Result directory already exists: {output_directory}")
        

    # Save the model configuration to this output directory
    with open(model_config_file, "r") as f:  
        model_config_data = f.read()
    with open(os.path.join(output_directory, os.path.basename(model_config_file)), 'w') as f:
        f.write(model_config_data)

    return output_directory
    
# Saving the results

def save_priors(model, model_config, output_directory):
    # Generate a prior sample from the model and save it for diagnostic purposes
    
    print("Saving priors...")
    
    prior_sample = model.generate(model_config["n_sims"])
    with open(f"{output_directory}prior_sample.npy", 'wb') as f:
        np.save(f, prior_sample)

def weight_posterior(posterior_sample, posterior_weights):
    """ Get the weighted joint posterior from unweighted posterior samples and ABC-SMC weights.

    Args:
        posterior_sample: posterior sample from result.samples.
        posterior_weights: ABC-SMC weights from result.weights.

    Returns:
        weighted_posterior (DataFrame): weighted posterior as a data frame.

    """

    try:
        posterior_sample_df = pd.DataFrame({"par1":posterior_sample["par1"], "par2":posterior_sample["par2"], "Dt":posterior_sample["Dt"]})
    except KeyError:
        print("Delay parameter not included.")
        posterior_sample_df = pd.DataFrame({"par1":posterior_sample["par1"], "par2":posterior_sample["par2"]})
    
    indx = np.arange(0, posterior_sample_df.shape[0], 1)
    weighted_posterior_indx = np.random.choice(indx, size = len(indx), p = posterior_weights/np.sum(posterior_weights))
    weighted_posterior = posterior_sample_df.iloc[weighted_posterior_indx]
    
    return weighted_posterior

def save_results(result, model, model_config, output_directory):
    """ Save the ELFI model results after inference.

    Args:
        result: ABC-SMC inference algorithm output.
        model (elfi.Model): ELFI model.
        model_config (dict): model configuration.
        output_directory (str): directory to save the results to.

    """

    print("Saving results...")
    model.save(output_directory)
    
    # Save the unweighted posterior samples & weights
    with open(f"{output_directory}posterior_abc_weights.npy", 'wb') as f:
        np.save(f, result.weights)
    with open(f"{output_directory}posterior_sample_uw.npy", 'wb') as f:
        np.save(f, result.samples)
        
    # Save discrepancies
    with open(f"{output_directory}discrepancies.npy", 'wb') as f:
        np.save(f, result.discrepancies)
    
    # Save predictive samples with the weighted posterior
    np.random.seed(model_config["seed"])
    weighted_result_samples = weight_posterior(result.samples, result.weights)
    
    # Save the weighted posterior samples & weights
    with open(f"{output_directory}posterior_sample_w.npy", 'wb') as f:
        ordered_dict = OrderedDict(weighted_result_samples.to_dict(orient='series'))
        np.save(f, ordered_dict)
    
    if not model_config["Dt_value"] == 0: # Dt included in parameters
    
        res_dict = model.generate(with_values = {'par1':np.array(weighted_result_samples["par1"]), 'par2':np.array(weighted_result_samples["par2"]), "Dt":np.array(weighted_result_samples["Dt"])}, outputs = ["SIRsim", "BSI", "yearly_BSI"])
    
    else:
        res_dict = model.generate(with_values = {'par1':np.array(weighted_result_samples["par1"]), 'par2':np.array(weighted_result_samples["par2"])}, outputs = ["SIRsim", "BSI", "yearly_BSI"])
    
    
    with open(f"{output_directory}ppred_sample.npy", 'wb') as f:
        np.save(f, res_dict)

    # Save the entire result -object
    with open(f"{output_directory}result.npy", 'wb') as f:
        np.save(f, result)
        
    
    # Save the graph of the model
    #g = elfi.draw(model, internal=False, param_names=False, filename=f"{output_directory}/elfi_graph", format="pdf")

    print(f"Results saved in {output_directory}")


# Visualisation functions

def plot_elfi(output_directory, model_config):
    """Create various preliminary visualizations using matplotlib from the results. Saves the results as .csv files for visualization with R.

    Args:
        output_directory (str): where to find the results of a run.
        model_config (dict): model configuration as a dictionary.

    """
    # Create a new directory for results
    vis_dir = f'{output_directory}vis/'
    try:
        os.makedirs(vis_dir) # separate folder for vis
    except FileExistsError:
        print("vis/ directory already exists")
        
    bsi_obs = get_incidence_data(model_config["bsi_data_file"], n_years = model_config["n_years"], clade = model_config['clade'], is_prop = model_config['is_prop'], n_incidence_pop = model_config['pop_size'])
    
    # Diagnostic visualisations:
    prior_sample = np.load(f'{output_directory}prior_sample.npy', allow_pickle = True)[()]
    posterior_sample = np.load(f'{output_directory}posterior_sample_w.npy', allow_pickle = True)[()]
    discrepancies = np.load(f'{output_directory}discrepancies.npy', allow_pickle = True)[()]
    pred_sample = np.load(f'{output_directory}ppred_sample.npy', allow_pickle = True)[()]
    posterior_weights = np.load(f'{output_directory}posterior_abc_weights.npy', allow_pickle = True)[()]
    
    # Save weighted posterior as a .csv file
    try:
        result_prior = pd.DataFrame({"par1":prior_sample["par1"], "par2":prior_sample["par2"], "Dt":prior_sample["Dt"]})
        result_posterior = weight_posterior(posterior_sample, posterior_weights)
        #pd.DataFrame({"par1":weight_posterior(posterior_sample, "par1")[c], "par2":weight_posterior(posterior_sample, "par2")[c], "Dt":weight_posterior(posterior_sample, "Dt")[c]})
    except KeyError:
        print("Dt not specified")
        result_prior = pd.DataFrame({"par1":prior_sample["par1"], "par2":prior_sample["par2"]})
        result_posterior = weight_posterior(posterior_sample, posterior_weights)
        #pd.DataFrame({"par1":weight_posterior(posterior_sample, "par1")[c], "par2":weight_posterior(posterior_sample, "par2")[c]})
    
    disc = pd.DataFrame({"discrepancies":discrepancies})
    SIR_I_node = pd.DataFrame(pred_sample["SIRsim"][1])
    BSI_yearly_node = pd.DataFrame(pred_sample["yearly_BSI"])

    try:
        os.makedirs(f"{output_directory}/csvs") # separate folder for vis
    except FileExistsError:
        print("csvs/ directory already exists")
        
    SIR_I_node.to_csv(f"{output_directory}/csvs/SIR_I_node.csv", index = False)
    BSI_yearly_node.to_csv(f"{output_directory}/csvs/BSI_yearly_node.csv", index = False)
    result_prior.to_csv(f"{output_directory}/csvs/result_prior.csv", index = False)
    result_posterior.to_csv(f"{output_directory}/csvs/result_samples.csv", index = False)
    disc.to_csv(f"{output_directory}/csvs/result_discrepancies.csv", index = False)
    
    ## Prior visualisations ##
    
    prior_directory = f"{vis_dir}prior/"
    try:
        os.makedirs(prior_directory)
    except FileExistsError:
        print("vis/prior/ already exists.")
        
    plot_priors_elfi(model_config["variables"], prior_sample, prior_directory, param1_name = model_config["param1"], param2_name = model_config["param2"]) # prior sample, variables of interest
    plot_prior_pred(prior_sample, prior_directory) # prior predictive figures

    print("Generating summary figures...") # plot histograms and pair plots of summaries of interest
    possible_summaries = ["reg_BSI_0", "reg_BSI_1", "log_BSI_max", "BSI_max_t", "log_BSI_max_prev", "log_BSI_max_next", "BSI_max", "BSI_max_prev", "BSI_max_next", "BSI_t0"]
    for i in range(len(possible_summaries)):
        for j in range(i+1, len(possible_summaries)):
            plot_4x4_summaries(possible_summaries[i], possible_summaries[j], prior_sample, prior_directory, fname = f"4x4_{possible_summaries[i]}_{possible_summaries[j]}_plot.pdf")
    print("Done!")


    ## Posterior (predictive) visualisations ##

    post_directory = f"{vis_dir}post/"
    print(post_directory)
    try:
        os.makedirs(post_directory)
    except FileExistsError:
        print("vis/post/ already exists.")
    
    plot_marginals(posterior_sample, post_directory)
    plot_discrepancy(posterior_sample, discrepancies, post_directory, param1 = model_config["param1"], param2 = model_config["param2"], clade = model_config["clade"])
    
    if model_config["clade"] == "B":
        max_dt = 6.5
    else:
        max_dt = 0.5
        
    plot_post_col_w_years(model_config["clade"], pred_sample, max_dt, vis_dir, save_fig = True) # plot posterior predictive SIR (colonisation) curves
    plot_post_col_w_CIs(model_config["clade"], pred_sample, vis_dir, save_fig = True) # plot pp colonization curves w 50/95CIs.
    
    plt.plot(pred_sample["BSI"].T, alpha = 0.5) # plot predicted weekly BSI cases
    plt.title("Predicted weekly BSI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"pred_weekly_BSI.pdf"), format="pdf", bbox_inches="tight")
    plt.clf()
    
    plt.plot(pred_sample["yearly_BSI"].T, alpha = 0.2, color = "grey") # plot yearly predicted BSI cases
    plt.plot(np.array(bsi_obs), label = "Observed BSI", marker = '*', linestyle = '--', color = "yellow")
    plt.title("Predicted yearly BSI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"pred_yearly_BSI.pdf"), format="pdf", bbox_inches="tight")
    plt.clf()
    
    visualize_ppc(pred_sample["yearly_BSI"], bsi_obs, output_directory = vis_dir, clade = model_config["clade"]) # plot comparison of observed data with predicted yearly BSI incidence

    ### Diagnostics & misc plots ##

    plot_diagnostics(posterior_sample, discrepancies, model_config["reparam"], vis_dir) # various diagnostic plots, R0 histogram, etc. See vis.py
    
    # prior sensitivity analysis
    print(f"{model_config['Dt_value']}")
    variables = model_config["variables"]
    if model_config["Dt_value"] == 0:
        variables.remove("Dt")
        plot_prior_sensitivity(prior_sample, posterior_sample, variables, var_names = [model_config["param1"], model_config["param2"]], save_fig = True, output_directory = vis_dir)
    else:
        plot_prior_sensitivity(prior_sample, posterior_sample, variables, var_names = [model_config["param1"], model_config["param2"], "Dt"], save_fig = True, output_directory = vis_dir)
    
    # convergence of discrepancies
    plot_convergence_of_threshold_pops(model_config["clade"], output_directory)




#### EXTRA ####
# If you want to create result csvs direclty from result.npy, try these.

def get_pop_df(result_populations, result_weights, thresholds, output_directory):
    """ Get parameters for all populations, annotated by treshold. Use this function if your results are presented as a dictionary.

    Args:
        result_populations (np.array): numpy array containing samples from each threshold population, stored in a dictionary.
        result_weights (np.array): ABC-SMC weights for each population.
        thresholds (list): thresholds for the populations.
        output_directory (str): the name of the directory.

    """

    p = len(result_populations)

    for i in range(0, p):
        
        pop_weights = result_weights[i]
        N = len(pop_weights)
        pop = weight_posterior(result_populations[i], pop_weights) # remember to weight the posterior!
        eps = np.repeat(thresholds[i], N)

        try:
            temp_df = pd.DataFrame({"par1":pop["par1"], "par2":pop["par2"], "Dt":pop["Dt"], "eps":eps})
        except KeyError:
            print("Dt not included.")
            temp_df = pd.DataFrame({"par1":pop["par1"], "par2":pop["par2"], "eps":eps})
        if i == 0:
            pop_df = temp_df
        else:
            pop_df = pd.concat([pop_df, temp_df])

                                  

    # Create a new directory for results
    csv_dir = f'{output_directory}csvs/'
    try:
        os.makedirs(csv_dir) # separate folder for vis
    except FileExistsError:
        print("csvs/ directory already exists")
        
    pop_df.to_csv(f"{output_directory}/csvs/result_pop_samples.csv", index = False)


def save_csvs(model, discrepancies, result_samples, result_weights, output_directory, seed):
    """ Save results stored in a dictionary to .csv files for visualization.

    Args:
        model (elfi.Model): model of interest
        discrepancies (numpy.ndarray): numpy array containing discrepancies
        result_samples (dict): dictionary containing the posterior samples for all parameters of interest (par1, par2, Dt)
        result_weights (numpy.ndarray): ABC-SMC weights.
        output_directory (str): directory to save the csvs to.
        seed (int): random seed for numpy.

    """
    
    np.random.seed(seed)
    weighted_result_samples = weight_posterior(result_samples, result_weights)
    print("saved weighted posterior")

    # generate a prior sample
    n_sims = len(result_samples["par1"])
    prior_sample = model.generate(n_sims)
    
    try:
        result_prior = pd.DataFrame({"par1":prior_sample["par1"], "par2":prior_sample["par2"], "Dt":prior_sample["Dt"]})
    except KeyError:
        result_prior = pd.DataFrame({"par1":prior_sample["par1"], "par2":prior_sample["par2"]})
    
    try: # Dt included in parameters
        res_dict = model.generate(with_values = {'par1':np.array(weighted_result_samples["par1"]), 'par2':np.array(weighted_result_samples["par2"]), "Dt":np.array(weighted_result_samples["Dt"])}, outputs = ["SIRsim", "BSI", "yearly_BSI"])
    except KeyError:
        res_dict = model.generate(with_values = {'par1':np.array(weighted_result_samples["par1"]), 'par2':np.array(weighted_result_samples["par2"])}, outputs = ["SIRsim", "BSI", "yearly_BSI"])

    print("generated res_dict")
    
    disc = pd.DataFrame({"discrepancies":discrepancies})
    SIR_I_node = pd.DataFrame(res_dict["SIRsim"][1])
    BSI_yearly_node = pd.DataFrame(res_dict["yearly_BSI"])

    # Create a new directory for results
    csv_dir = f'{output_directory}csvs/'
    try:
        os.makedirs(csv_dir) # separate folder for vis
    except FileExistsError:
        print("csvs/ directory already exists")

    SIR_I_node.to_csv(f"{output_directory}/csvs/SIR_I_node.csv", index = False)
    BSI_yearly_node.to_csv(f"{output_directory}/csvs/BSI_yearly_node.csv", index = False)
    result_prior.to_csv(f"{output_directory}/csvs/result_prior.csv", index = False)
    weighted_result_samples.to_csv(f"{output_directory}/csvs/result_samples.csv", index = False)
    disc.to_csv(f"{output_directory}/csvs/result_discrepancies.csv", index = False)


