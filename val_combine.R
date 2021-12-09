for (t in 1:100) {
    
    temp = list.files(path="~/rl/valresults", pattern=paste0("*_",t,".csv"), full.names=TRUE)
    files = lapply(temp, read.delim, header=TRUE, sep=",")

    file_mats = lapply(files, as.matrix)
    #file_array = simplify2array(file_mats)
    stack_mat = do.call(rbind, file_mats)
    #file_means = apply(file_array, c(1,2), mean)
    write.csv(stack_mat, file=paste0("~/rl/valcombresults/valcombresults_",t,".csv"), row.names=TRUE)

}

