#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

conda activate ecoli-elfi

# Synthetic data, not aggregated:

cp grid_params/grid_params.py grid_params.py
python3 run_grid.py

# Synthetic data, aggregated:

cp grid_params/grid_params_agg.py grid_params.py
python3 run_grid.py

# Reparam synthetic data:

cp grid_params/grid_params_rp.py grid_params.py
python3 run_grid.py


# Reparam synthetic data, aggregated:

cp grid_params/grid_params_agg_rp.py grid_params.py
python3 run_grid.py

# Observed data:

cp grid_params/grid_params_obs.py grid_params.py
python3 run_grid.py

# Observed data, reparametrized:

cp grid_params/grid_params_obs_rp.py grid_params.py
python3 run_grid.py