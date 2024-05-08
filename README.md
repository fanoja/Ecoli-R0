# Estimating R0 for Ecoli

Code for the estimation of R0 for *E. coli* colonisation.
- Model: SIR + observational model
- Method: likelihood-free inference

## Structure

- `doc/`: environment.yml file can be found here
- `data/`: datasets
- `cluster/`: for running the model on the command line (or on a computing cluster). In progress
- The main notebooks are `Clade A.ipynb` and `Clade C2.ipynb` for running the model and `visualization.ipynb` for result figures.

### Other relevant notebooks

- `Priors.ipynb`: Identifiability and testing different priors.
- `summaries.ipynb`: Testing summaries for ELFI.
- `diagnostics.ipynb`: Validation and diagnostics related code.
- `exploratory.ipynb`: Exploratory figures (replicates of previous studies and some original figures also).
- `SIR models.ipynb`: Notebook with a simple SIR model and least squares estimation (for practice). 
- `reparam SIR.ipynb`: Reparametrized model with net transmission and R as the parameters to estimate.
