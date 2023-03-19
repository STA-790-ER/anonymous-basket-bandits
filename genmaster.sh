#!/bin/bash

rm -rf ~/genslurmout/
mkdir ~/genslurmout
rm -rf ~/rl/genarrayresults/
mkdir ~/rl/genarrayresults
rm -rf ~/rl/genarrayselectionresults/
mkdir ~/rl/genarrayselectionresults
rm -f ~/rl/gencombresults/*.csv

sbatch ~/rl/lawson-bandits/genpar.sh
#wait

#module load R
#Rscipt ~/combine.R
