# test operators with pd matrix types
using PDMats
using Test

@testset "Scalar multiplication" begin
    pm1 = PDMat(eye(3))
    pm2 = PDiagMat(ones(3))
    pm3 = ScalMat(3,1)

    pm1a = PDMat(3.0 .* eye(3))
    pm2a = PDiagMat(3.0 .* ones(3))
    pm3a = ScalMat(3, 3)

    pmats = Any[pm1, pm2, pm3]
    pmatsa= Any[pm1a,pm2a,pm3a]

    for i in 1:length(pmats)
        @test Matrix(3.0 * pmats[i]) == Matrix(pmatsa[i])
        @test Matrix(pmats[i] * 3.0) == Matrix(pmatsa[i])
        @test Matrix(3 * pmats[i])   == Matrix(pmatsa[i])
        @test Matrix(pmats[i] * 3)   == Matrix(pmatsa[i])
    end
end
