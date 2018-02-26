__precompile__()

module PDMats

    using LinearAlgebra
    using SparseArrays
    using SuiteSparse

    import Base: +, *, \, /, ==, convert
    import LinearAlgebra: logdet, inv, diag, diagm, eigmax, eigmin, Cholesky
    import LinearAlgebra.BLAS: nrm2, axpy!, gemv!, gemm, gemm!, trmv, trmv!, trmm, trmm!
    import LinearAlgebra.LAPACK: trtrs!

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

    include("chol.jl")   # make Cholesky compatible with both Julia 0.3 & 0.4
    include("utils.jl")

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
