module PDMatsSparseArraysExt

using PDMats
using SparseArrays

using PDMats.LinearAlgebra

if isdefined(Base, :get_extension)
    const HAVE_CHOLMOD = isdefined(SparseArrays, :CHOLMOD)
else
    import SuiteSparse
    const HAVE_CHOLMOD = isdefined(SuiteSparse, :CHOLMOD)
end

if HAVE_CHOLMOD
    include("chol.jl")
    include("pdsparsemat.jl")
end

end # module
