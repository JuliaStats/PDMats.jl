__precompile__()

module PDMats

    using Compat
    using LinearAlgebra, SparseArrays

    import Base: +, *, \, /, ==, convert, inv, Matrix
    import LinearAlgebra: logdet, diag, eigmax, eigmin

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

#    import Base.BLAS: nrm2, axpy!, gemv!, gemm, gemm!, trmv, trmv!, trmm, trmm!
#    import Base.LAPACK: trtrs!
    import LinearAlgebra: A_ldiv_B!, A_mul_B!, A_mul_Bc!, A_rdiv_B!, A_rdiv_Bc!, Ac_ldiv_B!


    # The abstract base type

    abstract type AbstractPDMat{T<:Real} end

    # source files

    include("chol.jl")   # make Cholesky compatible with both Julia 0.3 & 0.4
    include("utils.jl")

    include("pdmat.jl")
#=
    if isdefined(Base.SparseArrays, :CHOLMOD)
        include("pdsparsemat.jl")
    end
=#
    include("pdiagmat.jl")
    include("scalmat.jl")

    include("generics.jl")
    include("addition.jl")

    include("testutils.jl")

    include("deprecates.jl")

end # module
