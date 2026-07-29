using PDMats
using LinearAlgebra
using SparseArrays
using Test

# Before loading StatsBase
@test Base.get_extension(PDMats, :StatsBaseExt) === nothing

# After loading StatsBase
using StatsBase
@test Base.get_extension(PDMats, :StatsBaseExt) isa Module

# Minimal `AbstractPDMat` implementation for the `cor2cov`/`cov2cor` fallbacks; must be top-level
# since a `struct` cannot be defined inside a `@testset`. `ScalMat`, `PDiagMat` and `PDMat` all
# specialize `cov2cor`, so only an external type such as this one reaches its fallback. The
# fallbacks are defined in terms of `X_A_Xt`, which subtypes have to implement themselves.
struct WrappedPD{T <: Real, P <: AbstractPDMat{T}} <: AbstractPDMat{T}
    parent::P
end
Base.size(A::WrappedPD) = size(A.parent)
Base.getindex(A::WrappedPD, i::Int, j::Int) = A.parent[i, j]
PDMats.X_A_Xt(A::WrappedPD, x::AbstractMatrix) = X_A_Xt(A.parent, x)

@testset "cor2cov" begin
    for T in (Float64, Float32)
        σ = rand(T, 3)
        for S in (Float64, Float32)
            X = randn(S, 3, 3)
            R = cov2cor(X * X' + I)
            # `cor2cov` scales the stored Cholesky factors, which differs between the two `uplo`s
            for C in (ScalMat(3, S(1)), PDiagMat(ones(S, 3)), PDMat(R), PDMat(R, cholesky(Symmetric(R, :L))))
                D = cor2cov(C, σ)
                @test D isa AbstractPDMat{promote_type(T, S)}
                @test size(D) == (3, 3)
                @test Matrix(D) ≈ cor2cov(Matrix(C), σ)
                if C isa PDMat
                    @test cholesky(D).uplo == cholesky(C).uplo
                    @test Matrix(cholesky(D)) ≈ D.mat
                end
            end
        end
    end
end

@testset "cov2cor" begin
    for S in (Float64, Float32)
        X = randn(S, 3, 3)
        A = X * X' + I
        # `cov2cor` scales the stored Cholesky factors, which differs between the two `uplo`s
        for D in (ScalMat(3, rand(S)), PDiagMat(rand(S, 3)), PDMat(A), PDMat(A, cholesky(Symmetric(A, :L))))
            for T in (Float64, Float32)
                σ = sqrt.(T.(diag(D)))
                rtol = sqrt(eps(T === Float64 && S === Float64 ? Float64 : Float32))
                C = cov2cor(D, σ)
                @test C isa AbstractPDMat{promote_type(T, S)}
                @test size(C) == (3, 3)
                @test Matrix(C) ≈ cov2cor(Matrix(D), σ) rtol = rtol
                if D isa PDMat
                    @test cholesky(C).uplo == cholesky(D).uplo
                    # `cov2cor` gives `mat` an exactly unit diagonal but scales the factors by `σ`,
                    # so the two only agree up to the accuracy of `σ`
                    @test Matrix(cholesky(C)) ≈ C.mat rtol = rtol
                end

                C = cov2cor(D)
                @test C isa AbstractPDMat{S}
                @test size(C) == (3, 3)
                @test Matrix(C) ≈ cov2cor(Matrix(D))
            end
        end
    end
end

# Sparse `PDMat`s are not backed by a `Cholesky`, so they convert the matrix and factorize anew
@testset "sparse PDMat" begin
    X = randn(3, 3)
    A = X * X' + I
    R = cov2cor(A)
    σ = rand(3)
    # `cor2cov` requires a correlation matrix: it overwrites the diagonal with `abs2.(σ)`
    C = cor2cov(PDMat(sparse(Symmetric(R))), σ)
    @test C isa AbstractPDMat{Float64}
    @test issparse(C.mat)
    @test Matrix(C) ≈ cor2cov(R, σ)
    D = PDMat(sparse(Symmetric(A)))
    E = cov2cor(D, sqrt.(diag(D)))
    @test E isa AbstractPDMat{Float64}
    @test issparse(E.mat)
    # `cov2cor` sets the diagonal instead of dividing, so it is exactly one
    @test diag(E) == ones(3)
    @test Matrix(E) ≈ cov2cor(A)
end

@testset "AbstractPDMat fallbacks" begin
    X = randn(3, 3)
    A = X * X' + I
    σ = sqrt.(diag(A))
    C = cov2cor(WrappedPD(PDMat(A)), σ)
    @test C isa AbstractPDMat{Float64}
    @test Matrix(C) ≈ cov2cor(A)

    R = cov2cor(A)
    b = rand(3)
    D = cor2cov(WrappedPD(PDMat(R)), b)
    @test D isa AbstractPDMat{Float64}
    @test Matrix(D) ≈ cor2cov(R, b)
end
