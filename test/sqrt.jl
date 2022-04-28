using PDMats
using Test
using LinearAlgebra: LinearAlgebra

_randPDSparseMat(T, n) = (X = T.(sprand(n, 1, 0.5)); PDSparseMat(X * X' + LinearAlgebra.I))

function _pd_sqrt_compare(A::AbstractPDMat)
    PDAsqrt = sqrt(A)
    PDAaqrt_dense = PDMat(sqrt(Matrix(A)))
    _pd_compare(PDAsqrt, PDAaqrt_dense)
end

function _pdsparse_sqrt_compare(A::AbstractPDMat)
    # specific method required for testing cholesky of a sparse matrix
    Asqrt = sqrt(A)
    Asqrt_dense = PDMat(sqrt(Matrix(A)))
    @test dim(Asqrt) == dim(Asqrt_dense)
    @test Matrix(Asqrt) ≈ Matrix(Asqrt_dense)
    cAsqrt = cholesky(Asqrt)
    perm = cAsqrt.p  # account for permutation
    L = sparse(cAsqrt.L)
    @test Asqrt[perm, perm] ≈ L * L'
end

n = 10

@testset "Matrix square root" begin
    for T in [Float32, Float64]
        _pd_sqrt_compare( _randPDMat(T, n))
        _pd_sqrt_compare( _randPDiagMat(T, n))
        _pd_sqrt_compare( _randScalMat(T, n))
        _pdsparse_sqrt_compare( _randPDSparseMat(T, n))
    end
end
