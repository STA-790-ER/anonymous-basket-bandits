temp = list.files(path="~/rl/genarrayresults", pattern=glob2rx("results*.csv"), full.names=TRUE)
agreetemp = list.files(path="~/rl/arrayresults", pattern = glob2rx("agree*.csv"), full.names = TRUE)
files = lapply(temp, read.delim, header=TRUE, sep=",")
gen_settings <- read.csv("~/rl/lawson-bandits/genset2.csv")[, -1]

if (length(agreetemp) > 0) {

agreefiles = lapply(agreetemp, read.delim, header = TRUE, sep = ",")
agree_mats = lapply(agreefiles, as.matrix)
agree_array = simplify2array(agree_mats)
agree_aggs = apply(agree_array, c(1,2), sum)
agree_prop = agree_aggs[1,1] / agree_aggs[1,2]
write.csv(agree_prop, file = "~/rl/combresults/agreeprop.csv")

}
print(nrow(gen_settings))
file_mats = lapply(files, as.matrix)
print(length(file_mats))
file_mat = do.call(rbind, file_mats)
policies <- colnames(file_mat)[-1]
print(length(unique(file_mat[,1])))
gen_settings_select <- gen_settings[file_mat[,1], ]
print(nrow(gen_settings_select))
full_mat <- cbind(gen_settings_select, file_mat[, -1])
file_mat <- file_mat[, -1]
#file_means = apply(file_array, c(1,2), mean)
#file_vars = apply(file_array, c(1,2), function(x) var(x) / length(x))

min_policies <- policies[apply(file_mat, 1, which.min)]
min_vals <- apply(file_mat, 1, min)
second_min_policies <- policies[apply(file_mat, 1, function(x) do.call(cbind, sort(x, index.return=TRUE))[2, 2])]
second_min_vals <- apply(file_mat, 1, function(x) sort(x)[2])
#print(apply(file_mat,1,which.min))
#print(min_policies)
full_df <- as.data.frame(full_mat)
full_df$min_policy <- min_policies
full_df$second_min_policy <- second_min_policies
full_df$min_val <- min_vals
full_df$second_min_val <- second_min_vals

library(tidyverse)

#print(agree_array)



write.csv(full_df %>% arrange(desc(second_min_val / min_val)), file="~/rl/gencombresults/gencombresults.csv", row.names=FALSE)
#write.csv(file_vars, file="~/rl/combresults/combvars.csv", row.names=TRUE)


#library(tidyverse)


#ploty <- file_means %>% as.data.frame() %>% mutate(Time = 1:nrow(file_means)) %>% pivot_longer(cols = colnames(file_means), names_to = "Policy", values_to = "Regret") %>%
#    ggplot(aes(x = Time, y = Regret, colour = Policy)) +
#    geom_line() +
#    labs(x = "Time", y = "Cumulative Expected Regret", title = "Regret by Policy")


#pdf(file="~/rl/combresults/regretplot.pdf")
#ploty
#dev.off()


print(full_df %>% head(100))

print(full_df %>% select(greedy_policy, thompson_policy, bayes_ucb_policy, ids_policy) %>% apply(2,sd))

