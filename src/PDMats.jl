__precompile__()

module PDMats

    using LinearAlgebra, SparseArrays, SuiteSparse

    import Base: +, *, \, /, ==, convert, inv, Matrix

    export
        # Types
        AbstractPDMat,
        PDMat,
        PDSparseMat,
        PDiagMat,
        ScalMat,

        # Functions
        dim,
        full,
        whiten,
        whiten!,
        unwhiten,
        unwhiten!,
        pdadd,
        pdadd!,
        add_scal,
        add_scal!,
        quad,
        quad!,
        invquad,
        invquad!,
        X_A_Xt,
        Xt_A_X,
        X_invA_Xt,
        Xt_invA_X,
        test_pdmat


    # The abstract base type

    abstract type AbstractPDMat{T<:Real} end

    const HAVE_CHOLMOD = isdefined(SuiteSparse, :CHOLMOD)

    # source files

    include("chol.jl")
    include("utils.jl")

    include("pdmat.jl")

    if HAVE_CHOLMOD
        include("pdsparsemat.jl")
    end

    include("pdiagmat.jl")
    include("scalmat.jl")

    include("generics.jl")
    include("addition.jl")

    include("testutils.jl")

    include("deprecates.jl")

end # module
