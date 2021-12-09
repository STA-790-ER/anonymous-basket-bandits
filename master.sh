#!/bin/bash

rm -f ~/slurmout/*
rm -f ~/rl/arrayresults/*.csv
rm -f ~/rl/combresults/*.csv

sbatch ~/rl/lawson-bandits/par.sh
#wait

#module load R
#Rscipt ~/combine.R
