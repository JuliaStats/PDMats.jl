using PDMats
using Test
using LinearAlgebra: LinearAlgebra

_randPDSparseMat(T, n) = (X = T.(sprand(n, 1, 0.5)); PDSparseMat(X * X' + LinearAlgebra.I))

function _pd_sqrt_compare(A::AbstractPDMat)
    PDAsqrt = sqrt(A)
    PDAaqrt_dense = PDMat(sqrt(Matrix(A)))
    _pd_compare(PDAsqrt, PDAaqrt_dense)
end

n = 10

@testset "Matrix square root" begin
    for T in [Float32, Float64]
        _pd_sqrt_compare( _randPDMat(T, n))
        _pd_sqrt_compare( _randPDiagMat(T, n))
        _pd_sqrt_compare( _randScalMat(T, n))
        _pd_sqrt_compare( _randPDSparseMat(T, n))
    end
end
