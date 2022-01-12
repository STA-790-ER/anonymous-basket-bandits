
t=Sys.getenv("SLURM_ARRAY_TASK_ID")    
temp = list.files(path="/hpc/group/laberlabs/jml165/truevalresults", pattern=paste0("*_",t,".csv"), full.names=TRUE)
files = lapply(temp, read.delim, header=TRUE, sep=",")
unlink(paste0("/hpc/group/laberlabs/jml165/truevalresults/*_",t,".csv"))
file_mats = lapply(files, as.matrix)
#file_array = simplify2array(file_mats)
stack_mat = do.call(rbind, file_mats)
#file_means = apply(file_array, c(1,2), mean)
write.table(stack_mat, file=paste0("/hpc/group/laberlabs/jml165/truevalcombresults/truevalcombresults_",t,".csv"), row.names=FALSE, sep=",")


