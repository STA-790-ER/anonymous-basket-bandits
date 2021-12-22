using Flux, CSV, Tables, Statistics
using BSON: @save
using Flux: @epochs
using Flux.Data: DataLoader
const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

    
dat = CSV.File("/hpc/home/jml165/rl/valcombresults/results_$(idx).csv") |> Tables.matrix

ncol = size(dat)[2]
nrow = size(dat)[1]
test_size = div(nrow, 10)

X_test = dat[1:test_size, 2:ncol]'
Y_test = dat[1:test_size, 1]'

X_train = dat[(test_size+1):nrow, 2:ncol]'
Y_train = dat[(test_size+1):nrow, 1]'

batch_data = DataLoader((X_train, Y_train), batchsize = 128, shuffle = true)

opt = Descent(.01)

hidden_size = div(ncol-1, 2) + 2

m = Chain(Dense(ncol - 1, hidden_size, σ), Dense(hidden_size, hidden_size, σ), Dense(hidden_size,div(hidden_size,2),σ), Dense(div(hidden_size,2), 1))
loss(x,y) = Flux.Losses.mse(m(x),y)
eval_loss(x,y) = mean(abs.((m(x) .- y) ./ y))
#evalcb() = @show(loss(X_test, Y_test))
evalcb() = @show([eval_loss(X_test, Y_test), loss(X_test, Y_test)])
#[(X_train,Y_train)],
@epochs 5 Flux.train!(loss, Flux.params(m), batch_data, opt, cb = evalcb)

@save "/hpc/home/jml165/rl/valneuralnets/valnn_$(idx).bson" m




