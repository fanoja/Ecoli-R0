import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

# Various functions for quick visualisation of results & diagnostics
# For the final, polished visualisations using R, please see visualisations.ipynb
        
def plot_priors_elfi(pars, prior_sample, output_directory = "res/elfi_res/test", param1_name = "par1", param2_name = "par2"):
    """ Plot a sample drawn from the elfi model priors for each parameter in prior_sample & save the figure.

    Args:
        pars (list): contains parameter names as strings. For example, pars = ["par1", "par2"]. Use the names that were used for inference.
        prior_sample (dict): contains a sample from the prior for parameters specified in pars.
        output_directory (string): file path to a directory where the resulting figure is saved.
        param1_name (string): name for the first parameter that you wish to use in the figure - usually more informative than in pars, for example "Net transmission".
        param2_name (string): name for the 2nd parameter that will be used in the resulting figure.

    """

    def check_for_parname(p):
        if p == "par1":
            pname = param1_name
        elif p == "par2":
            pname = param2_name
        else:
            pname = p
        return pname
    
    for par in pars:
        try:
            plt.hist(prior_sample[par])
            pname = check_for_parname(par)
            plt.title(pname)
            plt.savefig(os.path.join(output_directory, f"{pname}_prior_hist.pdf"), format="pdf", bbox_inches="tight")
            plt.clf()
        except KeyError:
            print(f"Par {par} not specified.")
    
    for p1 in pars:
        for p2 in pars:
            if p1 != p2:
                try:
                    plt.scatter(prior_sample[p1], prior_sample[p2])
    
                    p1_name = check_for_parname(p1)
                    p2_name = check_for_parname(p2)
                    
                    plt.title(f"{p1_name} and {p2_name}")
                    plt.savefig(os.path.join(output_directory, f"{p1_name}_{p2_name}_prior_scatter.pdf"), format="pdf", bbox_inches="tight")                    
                    plt.clf()
                except KeyError:
                    print(f"{par} not specified.")


def plot_prior_pred(prior_sample, output_directory = "res/elfi_res/test"):
    """Plot and save prior predictive curves for simulation draws, one curve per draw. Plots yearly colonization counts, weekly BSI counts and yearly BSI counts.

    Args:
        prior_sample (dict): contains the draws for the parameters of interest from the prior.
        output_directory (string): file path to the directory where the figure will be saved.
        
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(prior_sample["SIRsim"][1][0:200].T)
    axs[0, 0].set_title(f'SIR')
    axs[0, 1].plot(prior_sample["BSI"][0:200].T)
    axs[0, 1].set_title(f'Weekly BSI')
    axs[1, 0].plot(prior_sample["yearly_BSI"][0:200].T)
    axs[1, 0].set_title(f'Yearly BSI')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "node_draws.pdf"), format="pdf", bbox_inches="tight")
    plt.clf()


def plot_4x4_summaries(val1, val2, prior_sample, output_directory = "res/elfi_res/test", fname = "4x4_plot.pdf", debug = False):
    """Visualize prior draws from two summaries as a histogram of each and a joint scatterplot of summary values. Save the figure as pdf.

    Args:
        val1 (string): name of the first summary, as defined in the ELFI model.
        val2 (string): name of the second summary, as defined in the ELFI model.
        prior_sample (dict): contains summary draws from the prior ELFI model.
        output_directory (string): file path to the directory where the figure will be saved.
        fname (string): name of the resulting figure pdf.
        debug (bool): if True, prints out extra details.

    """
    if not val1 in prior_sample.keys():
        if debug:
            print(f"Summary {val1} not included")
    elif not val2 in prior_sample.keys():
        if debug:
            print(f"Summary {val2} not included")
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
        axs[0, 0].hist(prior_sample[f'{val1}'], bins = 20)
        axs[0, 0].set_ylabel(f'{val1}')
        axs[0, 1].scatter(prior_sample[f'{val2}'], prior_sample[f'{val1}'])
        axs[1, 0].scatter(prior_sample[f'{val1}'], prior_sample[f'{val2}'])
        axs[1, 0].set_ylabel(f'{val2}')
        axs[1, 0].set_xlabel(f'{val1}')
        #axs[1, 0].set_title(f'S2 and S1')
        axs[1, 1].hist(prior_sample[f'{val2}'], bins = 20)
        axs[1, 1].set_xlabel(f'{val2}')
        
        plt.savefig(os.path.join(output_directory, fname), format="pdf", bbox_inches="tight")
        plt.clf()

  
### Posterior visualisations

def plot_marginals(posterior_sample, output_directory = "res/elfi_res/test", save_fig = True):
    """Create a Seaborn pairplot from the posterior sample and save it as pdf.

    Args:
        posterior_sample (dict): contains the posterior sample for the parameters of the model.
        output_directory (str): path to the directory where the figure will be saved.
        save_fig (bool): if False, figure is not saved, only displayed.
        
    """
    pd_data = pd.DataFrame.from_dict(posterior_sample, orient='columns')
    ax = sns.pairplot(pd_data, kind="kde", diag_kind="kde")
    if save_fig:
        plt.savefig(os.path.join(output_directory, "marginals_posterior.pdf"), format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_discrepancy(posterior_sample, discrepancies, output_directory = "res/elfi_res/test", param1 = "beta", param2 = "gamma", clade = "NA"):
    """Create a discrepancy scatterplot for the clade of interest.

    Args:
        posterior_sample (dict): posterior sample from the parameters of interest.
        discrepancies: discrepancies
        output_directory (str): path to the directory where the figure will be saved.
        param1 (string): name of parameter 1.
        param2 (string): name of parameter 2.
        clade (string): clade of interest.


    """
    # Discrepancy scatterplot
    plt.scatter(posterior_sample['par1'], posterior_sample['par2'], c = discrepancies, alpha = 0.5, s = 0.5)
    if param1 == "beta" or param2 == "gamma":
        plt.title(fr"ST131-{clade}: Discrepancy for ($\{param1}$, $\{param2}$) -pairs")
        plt.xlabel(fr"$\{param1}$")
        plt.ylabel(fr"$\{param2}$")
    else:
        plt.title(f"ST131-{clade}: Discrepancy for ({param1}, {param2}) -pairs")
        plt.xlabel(f"{param1}")
        plt.ylabel(f"{param2}")
    plt.set_cmap('jet')
    plt.colorbar()
    plt.savefig(os.path.join(output_directory, f"discrepancy.pdf"), format="pdf", bbox_inches="tight")
    plt.clf()


# Posterior predictive check

def visualize_ppc(pred_BSI, bsi_obs_data, output_directory = "res/elfi_res/test", clade = "NA", ci = 95, save_fig = True):
    """Visualisation of the predicted BSI curve (mean, 95% CI) from the posterior estimates.

    Args:
        pred_BSI (numpy array): predicted BSI incidence, shape: (n_samples x n_years).
        bsi_obs_data (pandas Series): observed BSI incidence.
        output_directory (string): file path to a directory where the resulting figure is saved.
        clade (string): which clade to visualize, A or C2.
        ci (int): the credible intervals to use, by default 95%.
        save_fig (bool): if False, do not save the figure, only display it.

    """
    print(bsi_obs_data)
    expected_value = np.mean(pred_BSI, axis = 0)
    print(expected_value)
    med = np.median(pred_BSI, axis = 0)
    cis = np.percentile(pred_BSI, [(100-ci)/2, 100 - (100-ci)/2], axis = 0)
    t = np.arange(pred_BSI.shape[1])

    bsi_idx = list(bsi_obs_data.index)
    additional_labels = [max(bsi_idx) + i for i in range(1, len(t) - len(bsi_idx) + 1)]
    xtick_labels = bsi_idx + additional_labels
    print(xtick_labels)
    
    plt.plot(expected_value, label = "mean", color = "blue")
    plt.plot(med, label = "median", color = "lightblue")
    plt.plot(cis[0,:], label = f"Lower {ci}% CI", color = "gray")
    plt.plot(cis[1,:], label = f"Upper {ci}% CI", color = "gray")
    plt.plot(np.array(bsi_obs_data), label = "Observed BSI", marker = '*', linestyle = '--', color = "orange")
    plt.xticks(t, xtick_labels, rotation = 45)
    plt.legend()
    plt.title(f"Clade {clade} Pointwise Posterior Predictive")
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_directory, f"pred_obs_BSI.pdf"), format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_post_col_w_CIs(clade, pred_sample, output_directory = "res/elfi_res/test", save_fig = True):
    """Posterior predictive colonisation: mean, median 50/95% CIs"""

    pred_col = pred_sample["SIRsim"][1]
    expected_value = np.mean(pred_col, axis = 0)
    med = np.median(pred_col, axis = 0)
    cis95 = np.percentile(pred_col, [(100 - 95)/2, 100 - (100 - 95)/2], axis = 0)
    cis50 = np.percentile(pred_col, [(100 - 50)/2, 100 - (100 - 50)/2], axis = 0)
    t = np.arange(pred_col.shape[1])

    plt.plot(expected_value, label = "mean", color = "blue")
    plt.plot(med, label = "median", color = "lightblue")
    plt.plot(cis50[0,:], label = f"Lower {50}% CI", color = "darkgray")
    plt.plot(cis50[1,:], label = f"Upper {50}% CI", color = "darkgray")
    plt.plot(cis95[0,:], label = f"Lower 95% CI", color = "gray")
    plt.plot(cis95[1,:], label = f"Upper 95% CI", color = "gray")
    plt.legend()
    plt.title(f"Clade {clade} Pointwise Posterior Predictive Colonization")
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_directory, f"ppred_col_CIs.pdf"), format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_post_col_w_years(clade, pred_sample, max_delay_in_years = 0.5, output_directory = "res/elfi_res/test", save_fig = True):
    """Posterior predictive colonisation with years as xlabels.

    Args: 
        clade (string): clade of interest.
        pred_sample (dict): predicted colonization (etc.).
        output_directory (str): path to the directory where the figure will be saved.
        save_fig (bool): if False, displays the figure without saving.

    """
    n_years = 14

    if clade == "A":
        start_year_obs = 2004
    elif clade == "B":
        start_year_obs = 2010
        n_years -= 6
    else:
        start_year_obs = 2007
        n_years -= 3

    start_year_sim = int(start_year_obs) - max_delay_in_years
    n_simulated_weeks = int((n_years + max_delay_in_years)*52) # could also get this from pred_sample
    
    xtick_weeks = []
    current_week = 0
    
    while n_simulated_weeks - current_week >= 52:
        xtick_weeks.append(current_week)
        current_week = current_week + 52
    
    n_xticks = len(xtick_weeks)
    xtick_years = [start_year_sim + i for i in range(0, n_xticks)]
    #t = [n*52 for n in range(0, n_years)]
    #xtick_labels = [start_year - dt + years for years in range(0, dt)] + [start_year + years for years in range(0, n_years)]
    plt.plot(pred_sample["SIRsim"][1].T, alpha = 0.5, color = "grey")
    plt.title("Predicted colonization (weekly)")
    plt.xticks(xtick_weeks, xtick_years, rotation = 90)
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_directory, f"ppred_col_w_years.pdf"), format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_ppc_with_eps(pred_sample, discrepancies, clade, bsi_obs_data, output_directory = "res/elfi_res/test", eps = 1.0, save_fig = True):
    """Plot the posterior predictive figure with a threshold.
    
    Args:
        pred_sample (dict):  predicted colonization (etc.).
        discrepancies: discrepancies for each posterior draw.
        clade (string): clade of interest, A or C2.
        bsi_obs_data (Pandas Series): observed BSI incidence.
        output_directory (str): path to the directory where the figure will be saved.
        eps (int or float): threshold below which the predicted yearly BSI is visualized.
        save_fig (bool): if False, displays the figure without saving
        
    """
    
    yearly_BSI_eps = pred_sample["yearly_BSI"][np.where(discrepancies <= eps)]
    visualize_ppc(yearly_BSI_eps, bsi_obs_data, output_directory = output_directory, clade = clade, ci = 95, save_fig = save_fig)

def plot_diagnostics(posterior_sample, discrepancies, reparam, output_directory = "res/elfi_res/test", eps = None, save_fig = True):
    """Plots a series of misc figures from the posterior.

    Args:
        posterior_sample (dict): contains the posterior draws for parameters of interest.
        discrepancies: discrepancies for each posterior draw.
        reparam (bool): if True, uses the model with net transmission and basic reproduction rate (reproductive value).
        output_directory (str): path to the directory where the figure will be saved.
        eps (int or float): if not None, plots only posterior draws with discrepancies below this threshold.
        save_fig (bool): if False, display the figure without saving.
    """
    
    ## Diagnostics
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs)+1)/float(len(xs))
        return xs, ys
    
    xs, ys = ecdf(discrepancies)
    plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
    if save_fig:
        plt.savefig(os.path.join(output_directory, f"ecdf.pdf"), format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
    
    # Discrepancy, posterior
    if eps == None:
        p1 = posterior_sample['par1']
        p2 = posterior_sample['par2']
    else:
        p1 = posterior_sample['par1'][discrepancies < eps]
        p2 = posterior_sample['par2'][discrepancies < eps]


    if reparam:
        print("Reparametrized model!")
        #p1 = p1*p2/(p2 - 1) # beta
        #p2 = p1/(p2 - 1) # gamma
        plt.hist(p2)
        plt.title("R")
        if save_fig:
            plt.savefig(os.path.join(output_directory, f"R0_hist.pdf"), format="pdf", bbox_inches="tight")
            plt.clf()
        else:
            plt.show()

        print(f"R0 mean: {np.mean(p2)}")
        print(f"R0 median: {np.median(p2)}")
        ci = 95
        cis = np.percentile(p2, [(100-ci)/2, 100 - (100-ci)/2], axis = 0)
        print(f"95% CIs: {cis[0]}, {cis[1]}")
        print("Mean colonisation time")
        print(1/np.mean(( p1/(p2 - 1)))) # The time which this clade colonises an individual on average.        
        
    else:
        plt.hist(p1/p2)
        plt.title("R0")
        if save_fig:
            plt.savefig(os.path.join(output_directory, f"R0_hist.pdf"), format="pdf", bbox_inches="tight")
            plt.clf()
        else:
            plt.show()
    
        print(f"R0 mean: {np.mean(p1/p2)}")
        print(f"R0 median: {np.median(p1/p2)}")
        ci = 95
        cis = np.percentile(p1/p2, [(100-ci)/2, 100 - (100-ci)/2], axis = 0)
        print(f"95% CIs: {cis[0]}, {cis[1]}")
        
        print("Mean colonisation time")
        print(1/np.mean(p2)) # The time which this clade colonises an individual on average.
        
    # Net transmission posterior:

    if not reparam:
        plt.hist(p1 - p2)
        plt.title("Net transmission = beta - gamma")
        if save_fig:
            plt.savefig(os.path.join(output_directory, f"net_transmission_hist.pdf"), format="pdf", bbox_inches="tight")
            plt.clf()
        else:
            plt.show()



def plot_prior_sensitivity(prior_sample, posterior_sample, variables = ["par1", "par2", "Dt"], var_names = ["par1", "par2", "Dt"], save_fig = True, output_directory = "res/elfi_res/test"):
    """Plot a set of two histograms: one figure for each parameter. One histogram is the prior, the other the posterior. The goal is to compare the prior distribution to the posterior distribution as a prior sensitivity analysis.

    Args:
        prior_sample (dict): draws from the prior distribution for the parameters of interest.
        posterior_sample (dict): draws from the posterior distribution.
        variables (list): names of the parameters of interest, as used in inference.
        save_fig (bool): if False, display the figure without saving.
        output_directory (str): path to the directory where the figure will be saved.

    """
    
    fig, axs = plt.subplots(1, len(variables), figsize = (10,5))
    for i in range(0, len(variables)):
        axs[i].hist(posterior_sample[variables[i]], alpha = 0.5, label = "Posterior")
        axs[i].hist(prior_sample[variables[i]], alpha = 0.5, label = "Prior")
        axs[i].set_title(var_names[i])
        if i == 0:
            axs[i].legend()
    if save_fig:
        plt.savefig(os.path.join(output_directory, f"prior_sensitivity_hist.pdf"), format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_joint_distr_by_eps(posterior_sample, discrepancies, eps = [2,1.5,1.0]):
    """Plot the joint density of par1 and par2 for different threshold values.

    Args:
        posterior_sample (dict): the posterior sample for variables of interest.
        discrepancies (list): discrepancies.
        eps (list): thresholds to use for each plot.

    """
    import seaborn as sns
    
    def get_eps_posterior(eps):
    
        p1 = posterior_sample["par1"][np.where(discrepancies < eps)]
        p2 = posterior_sample["par2"][np.where(discrepancies < eps)]
    
        return p1, p2

    dfs = []
    for e in eps:
        p1, p2 = get_eps_posterior(e)
        n = len(p1)
        dfs.append(pd.DataFrame({par1_name:p1, par2_name:p2, "eps <=":[e]*n}))
    
    data = pd.concat(dfs)
    
    
    sns.jointplot(data=data, x=par1_name, y=par2_name, kind="kde", hue="eps <=")


def get_max_col_vs_bsi(res):
    """Get the maximum number of colonization counts and the corresponding yearly BSI cases.

    Args:
        res (dict): contains posterior (or prior) predictive SIRsim, yearly_BSI outputs etc.

    Returns:
        x (numpy array): max number of colonisations, length: n_sims.
        y (numpy array): max number of yearly BSI cases, length: n_sims.
    """
    
    x = np.max(res["SIRsim"][1], axis = 1) # should be n_sims long array
    y = np.max(res["yearly_BSI"], axis = 1)

    return x,y

def plot_max_bsi_max_col_scatter(prior_sample, pred_sample, discrepancies, eps = 1.0, title = "NA"):
    """A scatterplot with the max colonisation number as the function of max bsi cases, with the color indicating where the pairs are from (prior, posterior) set eps to get the posterior for a smaller or larger discrepancy threshold.

    Args:
        prior_sample (dict): draws from the prior distribution for quantities of interest.
        pred_sample (dict): posterior predictive colonization counts, BSI incidence etc.
        discrepancies (list): discrepancies.
        eps (int or float): plot only for samples with discrepancies smaller or equal to this threshold.
        title (string): title for the figure.

    """
    
    
    x_prior,y_prior = get_max_col_vs_bsi(prior_sample) # prior
    x,y = get_max_col_vs_bsi(pred_sample) # posterior

    x_eps = np.max(pred_sample["SIRsim"][1][np.where(discrepancies <= eps)], axis = 1)
    y_eps = np.max(pred_sample["yearly_BSI"][np.where(discrepancies <= eps)], axis = 1)
    
    plt.plot(x_prior, y_prior, ".", label = "Prior", alpha = 0.3, color = "grey")
    plt.plot(x, y, ".", label = "Posterior", alpha = 0.9, color = "orange")
    plt.plot(x_eps, y_eps, ".", label = f"Posterior, eps <= {eps}", alpha = 1, color = "forestgreen")
    plt.title(f"{title}")
    plt.xlabel("Max number of (weekly) colonisations")
    plt.ylabel("Max yearly BSI")
    plt.legend()
    plt.show()


# discrepancies for convergence, python version


def weight_posterior(posterior_sample, posterior_weights):
    # Weighted joint posterior

    try:
        posterior_sample_df = pd.DataFrame({"par1":posterior_sample["par1"], "par2":posterior_sample["par2"], "Dt":posterior_sample["Dt"]})
    except KeyError:
        print("Delay parameter not included.")
        posterior_sample_df = pd.DataFrame({"par1":posterior_sample["par1"], "par2":posterior_sample["par2"]})
    
    indx = np.arange(0, posterior_sample_df.shape[0], 1)
    weighted_posterior_indx = np.random.choice(indx, size = len(indx), p = posterior_weights/np.sum(posterior_weights))
    weighted_posterior = posterior_sample_df.iloc[weighted_posterior_indx]
    
    return weighted_posterior
    

def get_posterior_df_w_pops(posterior_sample, weights, eps):

    result_posterior = weight_posterior(posterior_sample, weights)
    result_posterior["eps"] = np.repeat([eps], len(result_posterior["par1"]))
    
    return result_posterior

def plot_convergence_of_threshold_pops(clade, output_directory):
    """Visualize the convergence of ABC-SMC by plotting the discrepancies as a scatterplot for all threshold populations.

    Args:
        clade (string): clade of interest (A, C2, B or C1)
        output_directory (string): path to results for the clade of interest

    """
    
    res_elfi = np.load(f"{output_directory}result.npy", allow_pickle = True)[()]
    res_pops = res_elfi.populations # all threshold populations in a list
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    
    for p in range(0, len(res_pops)):
    
        pop = res_pops[p]
        
        eps = pop.threshold#thresholds[p]
        weights = pop.weights

        weighted_pop = weight_posterior(pop.samples, weights)
        par1 = weighted_pop["par1"]
        par2 = weighted_pop["par2"]
        #dt = pop.samples["Dt"]
    
        if p == 0:
            pop_df = get_posterior_df_w_pops(pop.samples, weights, eps)
        else:
            pop_df = pd.concat([pop_df, get_posterior_df_w_pops(pop.samples, weights, eps)])
        
        #sns.kdeplot(x = par1, y = par2, ax = axes[0,p], color = "#34a8eb", label = "Clade A")
        axes[p].scatter(par1, par2, s = 1, color = "#34a8eb", label = f"Clade {clade}")
        if p == 0:
            axes[p].set_title(f"ST131-{clade} Threshold: {eps}")
            #axes[0, 0].legend()
        else:
            axes[p].set_title(f"Threshold: {eps}")
        axes[p].tick_params(axis='x', rotation=45)
        axes[p].set_xlabel("Net transmission")
        axes[p].set_ylim(1, 3)
        axes[p].set_xlim(0, 0.1)


    axes[0].set_ylabel("R")
    #axes[1,0].set_ylabel("R")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"vis/population_convergence.pdf"), format="pdf", bbox_inches="tight")
    plt.clf()
    
    # For ggplot2:
    pop_df.to_csv(f"{output_directory}/csvs/result_pop_samples.csv", index = False)

    