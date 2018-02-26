using Test
using PDMats
using LinearAlgebra
using SparseArrays
using SuiteSparse

include("testutils.jl")

println("Running tests ...")
for t in ["pdmtypes", "addition", "generics"]
    println("* $t ")
    include("$t.jl")
end
