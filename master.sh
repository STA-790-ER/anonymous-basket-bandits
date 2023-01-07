#!/bin/bash

rm -r ~/slurmout/
rm -r ~/rl/arrayresults/
rm -r ~/rl/arrayselectionresults/
rm -f ~/rl/combresults/*.csv

mkdir ~/slurmout/
mkdir ~/rl/arrayresults/
mkdir ~/rl/arrayselectionresults/

sbatch ~/rl/lawson-bandits/par.sh
#wait

#module load R
#Rscipt ~/combine.R
