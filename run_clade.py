"""Main script for running the model. 

Runs ABC-SMC using data for the clade of interest, using the simulation parameters specified in sim_params.py. Saves the results as .npy files in a directory.

Usage: on the command line/terminal, write:
python run_clade.py A
python run_clade.py C2

It is recommended to use a computing cluster when running a large number of simulations.

To create csv files of the results for final visualization with R, run plot_elfi.py <path_to_result_directory/>
"""

import sys
import importlib
import os
from datetime import datetime

clade = sys.argv[1] # specify a clade of interest from the command line
print(f"Clade {clade}")

if clade not in ["A", "C2"]:
    print("Invalid clade. Choosing clade A.")
    clade = "A"

with open("sim_params.py", "r") as f: # add clade from command line arguments
    data = f.readlines()

data[0] = f"clade = '{clade}'\n" # assume that clade information is on the first line

with open("sim_params.py", "w") as f: # write the clade on the first line of sim_params.py
    f.writelines(data)

# Create a folder for results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_directory = f"res/elfi_res/{clade}_results/clade_{clade}_{timestamp}/"
os.makedirs(output_directory)
print(f"Created a result directory: {output_directory}")

# Import the ELFI model from elfi_model.py
print("Importing SIR elfi model...")
import elfi_model
importlib.reload(elfi_model)
from elfi_model import *  

# Write simulation params to a file in output_directory
with open("sim_params.py", "r") as f:  
    sim_pars_data = f.read()
with open(os.path.join(output_directory, "sim_params.py"), 'w') as f:
    f.write(sim_pars_data)

# Generate a prior sample from the model and save it for diagnostic purposes
prior_sample = m.generate(10000)
with open(f"{output_directory}prior_sample.npy", 'wb') as f:
    np.save(f, prior_sample)

###### Running SMC-ABC #######

smc = elfi.SMC(m['d'], batch_size = 1000, seed = 20170530)

# final results A & C2: smc.sample(10000, thresholds = [1.0, 0.5, 0.35, 0.1])

if clade == "A": # ST131-A
    result = smc.sample(10000, thresholds = [1.0, 0.5, 0.35, 0.1])
else: # ST131-C2
    result = smc.sample(10000, thresholds = [1.0, 0.5, 0.35, 0.1])
    
##############################

# Save the posterior samples & weights
with open(f"{output_directory}posterior_abc_weights.npy", 'wb') as f:
    np.save(f, result.weights)
with open(f"{output_directory}posterior_sample.npy", 'wb') as f:
    np.save(f, result.samples)
    
# Save discrepancies
with open(f"{output_directory}discrepancies.npy", 'wb') as f:
    np.save(f, result.discrepancies)

# Save predictive samples with the weighted posterior

def weight_posterior(posterior_sample, variable, posterior_weights):
    # Weighted posterior.
    weighted_posterior = np.random.choice(posterior_sample[variable], size = len(posterior_sample[variable]), p = posterior_weights/np.sum(posterior_weights))

    return weighted_posterior
    
if not no_Dt: # Dt included in parameters
    res_dict = m.generate(with_values = {'par1':weight_posterior(result.samples, "par1", result.weights), 'par2':weight_posterior(result.samples, "par2", result.weights), "Dt":weight_posterior(result.samples, "Dt", result.weights)}, outputs = ["SIRsim", "BSI", "yearly_BSI"])
else:
    res_dict = m.generate(with_values = {'par1':weight_posterior(result.samples, "par1", result.weights), 'par2':weight_posterior(result.samples, "par2", result.weights)}, outputs = ["SIRsim", "BSI", "yearly_BSI"])

with open(f"{output_directory}ppred_sample.npy", 'wb') as f:
    np.save(f, res_dict)

# Save the entire result -object
with open(f"{output_directory}result.npy", 'wb') as f:
    np.save(f, result)
    

# Save the graph of the model
g = elfi.draw(m, internal=False, param_names=False, filename=f"{output_directory}/elfi_graph", format="pdf")
