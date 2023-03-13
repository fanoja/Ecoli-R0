conda activate ecoli-elfi

python3 reparamSIR.py > "logs/output_$(date +%d%m%Y_%H%M%S).txt"
