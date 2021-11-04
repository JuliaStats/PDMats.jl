
# test operators with pd matrix types
using LinearAlgebra, PDMats
using Test

@testset "scalar multiplication" begin
    printstyled("Testing scalar multiplication\n"; color=:blue)
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
