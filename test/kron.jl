using PDMats
using Test

n = 4
m = 7

for T in [Float64, Float32]
    X = randn(T, n, n)
    Y = randn(T, m, m)
    A = X * X'
    B = Y * Y'
    AkB = kron(A, B)
    PDA = PDMat(A)
    PDB = PDMat(B)
    PDAkB1 = PDMat(AkB)
    PDAkB2 = kron(PDA, PDB)
    @test PDAkB1.dim == PDAkB2.dim
    @test PDAkB1.mat ≈ PDAkB2.mat
    @test PDAkB1.chol.L ≈ PDAkB2.chol.L
    @test PDAkB1.chol.U ≈ PDAkB2.chol.U
    @test Matrix(PDAkB2.chol) ≈ PDAkB2.mat
end
