# Estimating R0 for Ecoli

Analysis code for the manuscript: "Basic reproduction number for pandemic
Escherichia coli clones is comparable to typical pandemic viruses"

## Directories and Files

- `doc/`: environment.yml file can be found here for installing necessary packages to run this model.
- `data/`: datasets
- `res/´: result files and visualisations
- `cluster/scripts/`: .py files containing the SIR model, observation model and loading the incidence and odds ratio data.
- `Clade A.ipynb` and `Clade C2.ipynb`: Notebooks detailing how the model is run.
-  `visualization.ipynb`: R code for visualisation of the results.
- `grid_params_clade_A.py` and `grid_params_clade_C2.py`: These files contain the relevant simulation parameters for each clade of interest, such as the population size, number of weeks to simulate, hyperparameters etc.
- ´grid_params.py`: This file gets overwritten by the contents of the files above, depending which clade is chosen for running the model. Do not modify.
- `elfi_model.py`: Contains the definition of the ELFI model. This model is loaded in `Clade A.ipynb` or `Clade C2.ipynb`.



