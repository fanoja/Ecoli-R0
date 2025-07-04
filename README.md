# Estimating R0 for Ecoli

Analysis code for the estimation of $R_0$ for *E. coli* colonisation.

## Usage

To run the model on a command line, run `python run_model.py <model_config>.txt`, where `<model_config>.txt` is a text file containing model configuration parameters. For an example, see `model_config_A_SIR.txt`. Running this script will save the model, results and a series of visualizations for preliminary results and diagnostics.

To run the model in a Python IDE, run the following lines:

`from elfi_functions import get_model, run_model`

`model_config_file = "model_config_A_SIR.txt" # specify model configuration`

`model = get_model(model_config_file) # get model`

`result = run_model(model, model_config_file) # run model and save results`

If you want to avoid saving results, replace the final line with the following:

`result = run_model(model, model_config_file, save_model = False)`

Article visualisations are generated using the `visualisation.ipynb` notebook. Input the directories containing clade A and clade C2 results and run the notebook.

## Directories and Files

- `data/`: datasets
- `res/`: result files and preliminary visualisations (`res\elfi_res\`) and final article visualisations (`res\article_vis\`)
-  `visualization.ipynb`: R code for visualisation of the results.
- `<model_config_file_name>.txt`: This file contains the relevant simulation parameters of interest common to clades A and C2, such as the population size, number of weeks to simulate, hyperparameters etc. You can create multiple model configuration files and input them as arguments to `run_model.py`
- `elfi_functions.py`: Contains the definition of the ELFI model in `get_model()` as well as functions for saving and visualizing the results.
- `BSI_functions.py` and `SIR_functions.py`: Contain functions related to the observation model and the SIR simulation, respectively.
- `run_model.py`: Main file for running the simulation via the command line (or a computing cluster).

## Conda environment

This repository contains an `environment.yml` file for package management. Run the following commands:
- `module load anaconda`: Loads the Anaconda module
- `conda env create --name ecoli-elfi --file environment.yml`: Creates the environment and to load the necessary packages.
Other potentially useful commands:
- If ipykernel is not installed: `conda install -c anaconda ipykernel`
- To make the environment available as a Jupyter kernel: `python -m ipykernel install --user --name=ecoli-elfi`
In case of installation problems, please consult the conda documentation.

## Acknowledgements

The model is defined using the Engine for Likelihood-Free Inference (ELFI) see the [ELFI documentation](\url{https://elfi.readthedocs.io/en/latest/index.html}).


