using PDMats
using LinearAlgebra
using Test

# Before loading StatsBase
@test Base.get_extension(PDMats, :StatsBaseExt) === nothing

# After loading StatsBase
using StatsBase
@test Base.get_extension(PDMats, :StatsBaseExt) isa Module

@testset "cor2cov" begin
    for T in (Float64, Float32)
        σ = rand(T, 3)
        for S in (Float64, Float32)
            for C in (ScalMat(3, S(1)), PDiagMat(ones(S, 3)), PDMat((x -> cov2cor(x * x' + I))(randn(S, 3, 3))))
                D = cor2cov(C, σ)
                @test D isa AbstractPDMat{promote_type(T, S)}
                @test size(D) == (3, 3)
                @test Matrix(D) ≈ cor2cov(Matrix(C), σ)
            end
        end
    end
end

@testset "cov2cor" begin
    for S in (Float64, Float32)
        for D in (ScalMat(3, rand(S)), PDiagMat(rand(S, 3)), PDMat((x -> x * x' + I)(randn(S, 3, 3))))
            for T in (Float64, Float32)
                σ = sqrt.(T.(diag(D)))
                C = cov2cor(D, σ)
                @test C isa AbstractPDMat{promote_type(T, S)}
                @test size(C) == (3, 3)
                @test Matrix(C) ≈ cov2cor(Matrix(D), σ) rtol = sqrt(eps(T === Float64 && S === Float64 ? Float64 : Float32))

                C = cov2cor(D)
                @test C isa AbstractPDMat{S}
                @test size(C) == (3, 3)
                @test Matrix(C) ≈ cov2cor(Matrix(D))
            end
        end
    end
end
