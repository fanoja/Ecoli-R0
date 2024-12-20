# Estimating R0 for Ecoli

Analysis code for the estimation of $R_0$ for *E. coli* colonisation.

## Usage

To run the analysis, open a command line and run the `run_clade.py` file: `python run_clade.py <clade>`, where `<clade>` is replaced by the clade of interest, either A or C2. Note: using a computing cluster is recommended to run the analysis.

After obtaining a result directory after running `run_clade.py`, run the command `python plot_elfi.py <path_to_folder>`. This script generates preliminary result figures using matplotlib and .csv files necessary to generate the final visualizations with R.

Article visualisations can be generated using the `visualisation.ipynb` notebook. Input the directories containing clade A and clade C2 results and run the notebook.

## Directories and Files

- `data/`: datasets
- `res/`: result files and preliminary visualisations (`res\elfi_res\`) and final article visualisations (`res\article_vis\`)
-  `visualization.ipynb`: R code for visualisation of the results.
- `sim_params.py`: This file contains the relevant simulation parameters of interest common to clades A and C2, such as the population size, number of weeks to simulate, hyperparameters etc.
- `elfi_model.py`: Contains the definition of the ELFI model.
- `BSI_functions.py` and `SIR_functions.py`: Contain functions related to the observation model and the SIR simulation, respectively.
- `run_clade.py`: Main file for running the simulation via the command line (or a computing cluster).
- `plot_elfi.py`: Generates preliminary visualisations from ELFI results and saves results as .csv files for `visualisation.ipynb`. Run after `run_clade.py`.
- `load_data.py`: Loads and preprocesses data from `data\`.

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


