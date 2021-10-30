using PDMats
using Test
using LinearAlgebra: LinearAlgebra

_randPDMat(T, n) = (X = randn(T, n, n); PDMat(X * X' + LinearAlgebra.I))
_randPDiagMat(T, n) = PDiagMat(rand(T, n))
_randScalMat(T, n) = ScalMat(n, rand(T))
function _pd_compare(A::AbstractPDMat, B::AbstractPDMat)
    @test dim(A) == dim(B)
    @test Matrix(A) ≈ Matrix(B)
    @test cholesky(A).L ≈ cholesky(B).L
    @test cholesky(A).U ≈ cholesky(B).U
    nothing
end
function _pd_kron_compare(A::AbstractPDMat, B::AbstractPDMat)
    PDAkB1 = kron(A, B)
    PDAkB2 = PDMat( kron( Matrix(A), Matrix(B) ) )
    _pd_compare(PDAkB1, PDAkB2)
    nothing
end

n = 4
m = 7

for T in [Float32, Float64]
    _pd_kron_compare( _randPDMat(T, n),    _randPDMat(T, m) )
    _pd_kron_compare( _randPDiagMat(T, n), _randPDiagMat(T, m) )
    _pd_kron_compare( _randScalMat(T, n),  _randScalMat(T, m) )
    _pd_kron_compare( _randPDMat(T, n),    _randPDiagMat(T, m) )
    _pd_kron_compare( _randPDiagMat(T, m), _randPDMat(T, n) )
    _pd_kron_compare( _randPDMat(T, n),    _randScalMat(T, m) )
    _pd_kron_compare( _randScalMat(T, m),  _randPDMat(T, n) )
    _pd_kron_compare( _randPDiagMat(T, n), _randScalMat(T, m) )
    _pd_kron_compare( _randScalMat(T, m),  _randPDiagMat(T, n) )
end
