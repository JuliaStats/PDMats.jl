module PDMatsSparseArraysExt

using PDMats
using SparseArrays

using PDMats.LinearAlgebra

const HAVE_CHOLMOD = isdefined(SparseArrays, :CHOLMOD)

if HAVE_CHOLMOD
    include("chol.jl")
    include("pdsparsemat.jl")
end

end # module
