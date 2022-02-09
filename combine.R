temp = list.files(path="~/rl/arrayresults", pattern="*.csv", full.names=TRUE)
files = lapply(temp, read.delim, header=TRUE, sep=",")

file_mats = lapply(files, as.matrix)
file_array = simplify2array(file_mats)

file_means = apply(file_array, c(1,2), mean)
file_vars = apply(file_array, c(1,2), function(x) var(x) / length(x))
write.csv(file_means, file="~/rl/combresults/combresults.csv", row.names=TRUE)
write.csv(file_vars, file="~/rl/combresults/combvars.csv", row.names=TRUE)

