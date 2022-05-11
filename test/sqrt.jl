using PDMats
using Test
using LinearAlgebra: LinearAlgebra

function _pd_sqrt_compare(A::AbstractPDMat)
    PDAsqrt = sqrt(A)
    Asqrt_dense = sqrt(Matrix(A))
    pdtest_cmat(PDAsqrt, Asqrt_dense, false, 0)
    pdtest_diag(PDAsqrt, Asqrt_dense, false, 0)
    pdtest_scale(PDAsqrt, Asqrt_dense, 0)
    return PDAsqrt, Asqrt_dense
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
