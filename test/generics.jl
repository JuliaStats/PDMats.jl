# test operators with pd matrix types
using LinearAlgebra, SparseArrays, PDMats
using Test

@testset "scalar multiplication" begin
    printstyled("Testing scalar multiplication\n"; color = :blue)
    pm1 = PDMat(Matrix(1.0I, 3, 3))
    pm2 = PDiagMat(ones(3))
    pm3 = ScalMat(3, 1)

    pm1a = PDMat(Matrix(3.0I, 3, 3))
    pm2a = PDiagMat(3.0 .* ones(3))
    pm3a = ScalMat(3, 3)

    pmats = Any[pm1, pm2, pm3]
    pmatsa = Any[pm1a, pm2a, pm3a]

    for i in 1:length(pmats)
        @test Matrix(3.0 * pmats[i]) == Matrix(pmatsa[i])
        @test Matrix(pmats[i] * 3.0) == Matrix(pmatsa[i])
        @test Matrix(3 * pmats[i]) == Matrix(pmatsa[i])
        @test Matrix(pmats[i] * 3) == Matrix(pmatsa[i])
    end
end

# issue #121
@test isposdef(PDMat([1.0 0.0; 0.0 1.0]))
@test isposdef(PDiagMat([1.0, 1.0]))
@test isposdef(ScalMat(2, 3.0))

@testset "ldiv!(A, b)" begin
    printstyled("Testing ldiv!(A, b)\n"; color = :blue)
    for A in (PDMat([4.0 2.0; 2.0 3.0]), PDiagMat([4.0, 3.0]), ScalMat(2, 4.0) #=PDSparseMat(sparse([4.0 2.0; 2.0 3.0]))=#) # CHOLMOD.Factor is not supported by ldiv!
        b = [1.0, 2.0]
        x = copy(b)
        ldiv!(A, x)
        @test isapprox(A * x, b)
        B = [1.0 2.0; 3.0 4.0]
        X = copy(B)
        ldiv!(A, X)
        @test isapprox(A * X, B)
    end
end
