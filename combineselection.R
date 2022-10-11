
temp = list.files(path="~/rl/arrayselectionresults", pattern=glob2rx("selectionresults*.csv"), full.names=TRUE)
files = lapply(temp, read.delim, header=TRUE, sep=",")

#file_mats = lapply(files, as.matrix)
#file_array = simplify2array(file_mats)

#file_means = apply(file_array, c(1,2), mean)
#file_vars = apply(file_array, c(1,2), function(x) var(x) / length(x))
dat_stack <- do.call(rbind, files)

#print(agree_array)

#write.csv(file_means, file="~/rl/combresults/combresults.csv", row.names=TRUE)
#write.csv(file_vars, file="~/rl/combresults/combvars.csv", row.names=TRUE)


library(tidyverse)

aggy <- dat_stack %>% as.data.frame() %>% mutate(dummy = 1)

#print(head(aggy))

aggy <- aggy %>% pivot_wider(names_from = policy, values_from = dummy, values_fn = function(x) sum(x, na.rm = TRUE), values_fill = 0) %>%
    mutate(tot = rowSums(select(., -time)), greedy_proportion = greedy_policy / tot)

write.csv(aggy, file="~/rl/combselectionresults/combselectionresults.csv", row.names = F)

ploty <- aggy %>% filter(time < 100) %>%
    ggplot(aes(x = time, y = greedy_proportion)) +
    geom_line() +
    labs(x = "Time", y = "Greedy Policy Proportion", title = "Greedy-Thompson Basket Selection Proportion")


pdf(file="~/rl/combselectionresults/selectionplot.pdf")
ploty
dev.off()



print(aggy)
