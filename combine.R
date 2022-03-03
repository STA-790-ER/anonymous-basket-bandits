temp = list.files(path="~/rl/arrayresults", pattern=glob2rx("results*.csv"), full.names=TRUE)
agreetemp = list.files(path="~/rl/arrayresults", pattern = glob2rx("agree*.csv"), full.names = TRUE)
files = lapply(temp, read.delim, header=TRUE, sep=",")

agreefiles = lapply(agreetemp, read.delim, header = TRUE, sep = ",")

file_mats = lapply(files, as.matrix)
file_array = simplify2array(file_mats)

agree_mats = lapply(agreefiles, as.matrix)
agree_array = simplify2array(agree_mats)
file_means = apply(file_array, c(1,2), mean)
file_vars = apply(file_array, c(1,2), function(x) var(x) / length(x))


#print(agree_array)
agree_aggs = apply(agree_array, c(1,2), sum)
agree_prop = agree_aggs[1,1] / agree_aggs[1,2]

write.csv(file_means, file="~/rl/combresults/combresults.csv", row.names=TRUE)
write.csv(file_vars, file="~/rl/combresults/combvars.csv", row.names=TRUE)

write.csv(agree_prop, file = "~/rl/combresults/agreeprop.csv")
