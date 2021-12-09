using Flux, CSV, Tables
using BSON: @save

const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

    
dat = CSV.File("/hpc/home/jml165/rl/valcombresults/valcombresults_$(idx).csv") |> Tables.matrix
    
X = dat[:, 3:14]'
Y = dat[:, 2]'

opt = Descent(.01)

m = Chain(Dense(12,6,σ), Dense(6,1))
loss(x,y) = sum(Flux.Losses.mse(m(x),y))
Flux.train!(loss, Flux.params(m), [(X,Y)], opt)

@save "/hpc/home/jml165/rl/valneuralnets/valnn_$(idx).bson" m

