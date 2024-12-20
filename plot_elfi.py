"""Generate visualisations from ELFI results, mainly for diagnostic purposes. 

Creates a set of diagnostic and preliminary figures saved to <output_directory>/vis
Note: Also saves the .npy files as .csv files. The official figures are generated using R from these, see "visualisation.ipynb". Required to run before generating the visualizations in "visualisation.ipynb".

Usage:
python plot_elfi.py <path_to_output_directory_as_string>
"""

import sys
res_dir = sys.argv[1] # read the directory where the results of run_clade.py are stored from the command line

# Read the contents of the source file:
with open(f"{res_dir}/sim_params.py", 'r') as f:
    sim_pars = f.read()
# Process sim_pars as variables:
exec(sim_pars)
#from sim_params import *

import importlib
import vis
importlib.reload(vis)
from vis import *

## Open the relevant files ##

prior_sample = np.load(f'{res_dir}prior_sample.npy', allow_pickle = True)[()]
posterior_sample = np.load(f'{res_dir}posterior_sample.npy', allow_pickle = True)[()]
discrepancies = np.load(f'{res_dir}discrepancies.npy', allow_pickle = True)[()]
pred_sample = np.load(f'{res_dir}ppred_sample.npy', allow_pickle = True)[()]
posterior_weights = np.load(f'{res_dir}posterior_abc_weights.npy', allow_pickle = True)[()]

## Saving the .npy result files as csv files for final visualization with R ##
# Weight the posterior sample, since ELFI does not do this by default:

def weight_posterior(posterior_sample, variable):
    # Weighted posterior. Assumes that posterior_abc_weights.npy is loaded.
    weighted_posterior = np.random.choice(posterior_sample[variable], size = len(posterior_sample[variable]), p = posterior_weights/np.sum(posterior_weights))

    return weighted_posterior
    
with_eps = True # set to true, if you wish to limit the posterior under a certain threshold (here, 0.1)
if with_eps:
    eps = 0.1
    c = np.where(discrepancies <= eps)

    result_prior = pd.DataFrame({"par1":prior_sample["par1"], "par2":prior_sample["par2"], "Dt":prior_sample["Dt"]})
    #result_posterior = pd.DataFrame({"par1":posterior_sample["par1"][c], "par2":posterior_sample["par2"][c], "Dt":posterior_sample["Dt"][c]})
    
    result_posterior = pd.DataFrame({"par1":weight_posterior(posterior_sample, "par1")[c], "par2":weight_posterior(posterior_sample, "par2")[c], "Dt":weight_posterior(posterior_sample, "Dt")[c]})
    
    disc = pd.DataFrame({"discrepancies":discrepancies[c]})
    SIR_I_node = pd.DataFrame(pred_sample["SIRsim"][1][c])
    BSI_yearly_node = pd.DataFrame(pred_sample["yearly_BSI"][c])
else:
    result_prior = pd.DataFrame({"par1":prior_sample["par1"], "par2":prior_sample["par2"], "Dt":prior_sample["Dt"]})
    result_posterior = pd.DataFrame({"par1":posterior_sample["par1"], "par2":posterior_sample["par2"], "Dt":posterior_sample["Dt"]})
    disc = pd.DataFrame({"discrepancies":discrepancies})
    SIR_I_node = pd.DataFrame(pred_sample["SIRsim"][1])
    BSI_yearly_node = pd.DataFrame(pred_sample["yearly_BSI"])
try:
    os.mkdir(f"{res_dir}/csvs")
except FileExistsError:
    print("csvs/ already exists.")

SIR_I_node.to_csv(f"{res_dir}/csvs/SIR_I_node.csv", index = False)
BSI_yearly_node.to_csv(f"{res_dir}/csvs/BSI_yearly_node.csv", index = False)
result_prior.to_csv(f"{res_dir}/csvs/result_prior.csv", index = False)
result_posterior.to_csv(f"{res_dir}/csvs/result_samples.csv", index = False)
disc.to_csv(f"{res_dir}/csvs/result_discrepancies.csv", index = False)

### Creating diagnostic visualisations ###

from load_data import get_incidence_data # need observed data
print(f"plot_elfi.py clade: {clade}")

if clade == "A":
    bsi_obs_data = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = pop_size)
else:
    bsi_obs_data = get_incidence_data("data/NORM_incidence.csv", clade = clade, is_prop = is_prop, n_incidence_pop = pop_size, remove_C2_first = True)

# Define output directory:
output_directory = f'{res_dir}vis/'
try:
    os.makedirs(output_directory) # separate folder for vis
except FileExistsError:
    print("vis/ directory already exists")

## Prior visualisations ##

prior_directory = f"{output_directory}prior/"
try:
    os.makedirs(prior_directory)
except FileExistsError:
    print("vis/prior/ already exists.")
    
plot_priors_elfi(variables, prior_sample, prior_directory, param1_name = param1, param2_name = param2) # prior sample, variables of interest
plot_prior_pred(prior_sample, prior_directory) # prior predictive figures

print("Generating summary figures...") # plot histograms and pair plots of summaries of interest
possible_summaries = ["reg_BSI_0", "reg_BSI_1", "log_BSI_max", "BSI_max_t", "log_BSI_max_prev", "log_BSI_max_next", "BSI_max", "BSI_max_prev", "BSI_max_next", "BSI_t0"]
for i in range(len(possible_summaries)):
    for j in range(i+1, len(possible_summaries)):
        plot_4x4_summaries(possible_summaries[i], possible_summaries[j], prior_sample, prior_directory, fname = f"4x4_{possible_summaries[i]}_{possible_summaries[j]}_plot.pdf")
print("Done!")

## Posterior (predictive) visualisations ##

post_directory = f"{output_directory}post/"
print(post_directory)
try:
    os.makedirs(post_directory)
except FileExistsError:
    print("vis/post/ already exists.")

plot_marginals(posterior_sample, post_directory)
plot_discrepancy(posterior_sample, discrepancies, post_directory, param1 = param1, param2 = param2, clade = clade)

plot_post_SIR_w_years(clade, pred_sample, output_directory, save_fig = True) # plot posterior predictive SIR (colonisation) curves

plt.plot(pred_sample["BSI"].T, alpha = 0.5) # plot predicted weekly BSI cases
plt.title("Predicted weekly BSI")
plt.tight_layout()
plt.savefig(os.path.join(output_directory, f"pred_weekly_BSI.pdf"), format="pdf", bbox_inches="tight")
plt.clf()

plt.plot(pred_sample["yearly_BSI"].T, alpha = 0.2, color = "grey") # plot yearly predicted BSI cases
plt.plot(np.array(bsi_obs_data), label = "Observed BSI", marker = '*', linestyle = '--', color = "yellow")
plt.title("Predicted yearly BSI")
plt.tight_layout()
plt.savefig(os.path.join(output_directory, f"pred_yearly_BSI.pdf"), format="pdf", bbox_inches="tight")
plt.clf()

visualize_ppc(pred_sample["yearly_BSI"], bsi_obs_data, output_directory = output_directory, clade = clade) # plot comparison of observed data with predicted yearly BSI incidence

### Diagnostics & misc plots ##

plot_diagnostics(posterior_sample, discrepancies, reparam, output_directory) # various diagnostic plots, R0 histogram, etc. See vis.py

# prior sensitivity analysis
plot_prior_sensitivity(prior_sample, posterior_sample, variables, save_fig = True, output_directory = output_directory)
