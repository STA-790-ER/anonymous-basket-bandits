#!/bin/bash

rm -f ~/slurmout2/*
rm -f ~/rl/arrayresults2/*.csv
rm -f ~/rl/arrayselectionresults2/*.csv
rm -f ~/rl/combresults2/*.csv

sbatch ~/rl/lawson-bandits/par2.sh
#wait

#module load R
#Rscipt ~/combine.R
