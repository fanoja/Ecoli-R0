#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

conda activate ecoli-elfi

cp grid_params/grid_params_obs.py grid_params.py
python3 run_grid.py
