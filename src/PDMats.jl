module PDMats

    using LinearAlgebra, SparseArrays, SuiteSparse

    import Base: +, *, \, /, ==, convert, inv, Matrix, kron

    export
        # Types
        AbstractPDMat,
        PDMat,
        PDSparseMat,
        PDiagMat,
        ScalMat,

        # Functions
        dim,
        whiten,
        whiten!,
        unwhiten,
        unwhiten!,
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
    abstract type AbstractPDMat{T<:Real} <: AbstractMatrix{T} end

    const HAVE_CHOLMOD = isdefined(SuiteSparse, :CHOLMOD)

    # source files

    include("utils.jl")
    include("chol.jl")

    include("pdmat.jl")

    if HAVE_CHOLMOD
        include("pdsparsemat.jl")
    end

    include("pdiagmat.jl")
    include("scalmat.jl")

    include("generics.jl")
    include("addition.jl")

    include("deprecates.jl")

end # module
