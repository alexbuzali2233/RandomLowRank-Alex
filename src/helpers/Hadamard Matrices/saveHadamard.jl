using NPZ
using Hadamard
include("config.jl")
npzwrite(hadamardMatrixPath * "hadamard144.npy", hadamard(144))