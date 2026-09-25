module PDMats

using LinearAlgebra

import Base: +, *, \, /, ==, convert, inv, Matrix, kron

export
    # Types
    AbstractPDMat,
    PDMat,
    PDiagMat,
    ScalMat,

    # Functions
    dim,
    whiten,
    whiten!,
    invwhiten,
    invwhiten!,
    unwhiten,
    unwhiten!,
    invunwhiten,
    invunwhiten!,
    pdadd,
    pdadd!,
    quad,
    quad!,
    invquad,
    invquad!,
    X_A_Xt,
    Xt_A_X,
    X_invA_Xt,
    Xt_invA_X


"""
The base type for positive definite matrices.
"""
abstract type AbstractPDMat{T <: Real} <: AbstractMatrix{T} end

# source files

include("utils.jl")
include("chol.jl")

include("pdmat.jl")
include("pdiagmat.jl")
include("scalmat.jl")

include("generics.jl")
include("addition.jl")
include("congruence.jl")

include("deprecates.jl")

end # module
