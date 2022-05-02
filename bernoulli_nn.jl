using Flux, CSV, Tables, Statistics
using BSON: @save
using Flux: @epochs
using Flux.Data: DataLoader
const idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

    
dat = CSV.File("/hpc/group/laberlabs/jml165/valcombresults/valcombresults_$(idx).csv") |> Tables.matrix
#dat = CSV.File("/hpc/home/jml165/rl/valcombresults/valcombresults_$(idx).csv") |> Tables.matrix

stds = std.(eachcol(dat))
means = mean.(eachcol(dat))

CSV.write("/hpc/home/jml165/rl/bernneuralnetscales/bernscales_$(idx).csv", Tables.table([means stds]), writeheader = false)

dat = Flux.normalise(dat, dims = 1)
ncol = size(dat)[2]
nrow = size(dat)[1]




test_size = div(nrow, 10)

X_test = dat[1:test_size, 2:ncol]'
Y_test = dat[1:test_size, 1]'

X_train = dat[(test_size+1):nrow, 2:ncol]'
Y_train = dat[(test_size+1):nrow, 1]'

#print("\n")
#print(X_train)
#print("\n")
#print(Y_train)
#print("\n")

batch_data = DataLoader((X_train, Y_train), batchsize = 128, shuffle = true)

opt = ADAM(.005)

hidden_size = div(ncol-1, 2) * 2
hidden_size = (ncol - 1)
m = Chain(Dense(ncol - 1, hidden_size, σ),Dense(hidden_size,hidden_size,σ),Dense(hidden_size, 1))

#m = Chain(Dense(ncol - 1, hidden_size, σ), Dense(hidden_size, 1))
loss(x,y) = Flux.Losses.mse(m(x),y)
eval_loss(x,y) = mean(abs.((m(x) .- y) ./ y))
#evalcb() = @show(loss(X_test, Y_test))
evalcb() = @show([round(eval_loss(X_test, Y_test); digits=3), round(loss(X_test, Y_test); digits = 3),round(eval_loss(X_train, Y_train); digits=3), round(loss(X_train, Y_train); digits = 3)])
#[(X_train,Y_train)],
@epochs 200 Flux.train!(loss, Flux.params(m), batch_data, opt, cb = evalcb)

@save "/hpc/home/jml165/rl/bernvalneuralnets/bernvalnn_$(idx).bson" m




