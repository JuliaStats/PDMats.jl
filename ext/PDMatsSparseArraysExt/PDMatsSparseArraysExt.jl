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

# https://github.com/JuliaLang/julia/pull/29749
if VERSION < v"1.1.0-DEV.792"
    eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))
end

if HAVE_CHOLMOD
    include("chol.jl")
    include("pdsparsemat.jl")
end

end # module
