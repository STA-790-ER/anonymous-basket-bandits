using Flux, CSV, Tables
using BSON: @save

const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

    
dat = CSV.File("/hpc/home/jml165/rl/valcombresults/valcombresults_$(idx).csv") |> Tables.matrix

ncol = size(dat)[2]


X = dat[:, 3:ncol]'
Y = dat[:, 2]'

opt = Descent(.01)

hidden_size = div(ncol-2, 2)

m = Chain(Dense(ncol - 2, hidden_size, σ), Dense(hidden_size, 1))
loss(x,y) = sum(Flux.Losses.mse(m(x),y))
Flux.train!(loss, Flux.params(m), [(X,Y)], opt)

@save "/hpc/home/jml165/rl/valneuralnets/valnn_$(idx).bson" m

