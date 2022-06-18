using PDMats
using Test
using LinearAlgebra: LinearAlgebra

function _pd_kron_compare(A::AbstractPDMat, B::AbstractPDMat)
    PDAkB_kron = @inferred kron(A, B)
    PDAkB_dense = PDMat( kron( Matrix(A), Matrix(B) ) )
    _pd_compare(PDAkB_kron, PDAkB_dense)
end

n = 4
m = 7

@testset "Kronecker product" begin
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
end
