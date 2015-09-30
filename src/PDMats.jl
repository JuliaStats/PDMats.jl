VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module PDMats

    using ArrayViews
    using Compat

    import Base: +, *, \, /, ==
    import Base: full, logdet, inv, diag, diagm, scale, scale!, eigmax, eigmin

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

    import Base.BLAS: nrm2, axpy!, gemv!, gemm, gemm!, trmv, trmv!, trmm, trmm!
    import Base.LAPACK: trtrs!
    import Base.LinAlg: A_ldiv_B!, A_mul_B!, A_mul_Bc!, A_rdiv_B!, A_rdiv_Bc!, Ac_ldiv_B!, Cholesky


    # The abstract base type

    abstract AbstractPDMat

    # source files

    include("chol.jl")   # make Cholesky compatible with both Julia 0.3 & 0.4
    include("utils.jl")

    include("pdmat.jl")
    include("pdsparsemat.jl")
    include("pdiagmat.jl")
    include("scalmat.jl")

    include("generics.jl")
    include("addition.jl")

    include("testutils.jl")

    include("deprecates.jl")

end # module
