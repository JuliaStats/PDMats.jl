using PDMats, LinearAlgebra, SparseArrays, StatsBase, Test

@testset "Existing real storage converts to complex arrays ($R)" for R in (Float32, Float64)
    matrices = Any[PDMat(R[3 1; 1 2]), PDiagMat(R[2, 3]), ScalMat(2, R(2))]
    if PDMats.HAVE_CHOLMOD && R === Float64
        push!(matrices, PDSparseMat(sparse(R[3 1; 1 2])))
    end
    for a in matrices, C in (ComplexF32, ComplexF64), target in (AbstractMatrix{C}, AbstractArray{C})
        converted = convert(target, a)
        @test eltype(converted) === C
        @test converted isa AbstractMatrix{C}
        @test Matrix(converted) == C.(Matrix(a))
    end
    for C in (ComplexF32, ComplexF64)
        @test Matrix(convert(PDMat{C}, first(matrices))) == C.(Matrix(first(matrices)))
    end
end

@testset "Dense Hermitian PDMat ($T)" for T in (ComplexF32, ComplexF64)
    R = typeof(real(zero(T)))
    tolerance = 128eps(R)
    L = T[2 0 0; 1 + im 3 0; 0.5 - 0.5im 2 + im 1.5]
    A = L * L'
    U = L'
    X = T[1 + im 2 - im; -1 + 2im 0.5 + im; 2 - im -1 - im]
    x = X[:, 1]
    upper = cholesky(Hermitian(A, :U))
    lower = cholesky(Hermitian(A, :L))

    # Independent explicit LinearAlgebra factors/expressions are the oracles.
    @testset "constructors and algebra" begin
        for d in (
                PDMat(A), PDMat(Hermitian(A)), PDMat(upper), PDMat(lower),
                PDMat{T}(A), PDMat{T, Matrix{T}}(A, upper), PDMat(A, lower),
            )
            @test eltype(d) === T
            @test Matrix(d) ≈ A rtol = tolerance
            @test ishermitian(d)
            @test !issymmetric(d)
            @test isposdef(d)
            @test d \ X ≈ A \ X rtol = tolerance
            @test X' / d ≈ X' / A rtol = tolerance
            @test Matrix(inv(d)) ≈ inv(A) rtol = tolerance
            @test logdet(d) isa R
            @test logdet(d) ≈ 2sum(log, real.(diag(L))) rtol = tolerance
            @test eigmin(d) ≈ eigmin(Hermitian(A)) rtol = tolerance
            @test eigmax(d) ≈ eigmax(Hermitian(A)) rtol = tolerance
            @test Matrix(sqrt(d))^2 ≈ A rtol = tolerance
        end
        d = PDMat(A)
        @testset "complex dense row-vector division" begin
            for v in (x, real.(x)), wrap in (adjoint, transpose)
                row = wrap(v)
                quotient = row / d
                @test size(quotient) == (1, length(x))
                @test eltype(quotient) === T
                @test quotient ≈ row / A rtol = tolerance
                @test quotient * A ≈ row rtol = tolerance atol = tolerance
                @test_throws DimensionMismatch wrap(v[1:2]) / d
            end
        end
        @test eltype(convert(PDMat{ComplexF64}, d)) === ComplexF64
        @test Matrix(convert(AbstractMatrix{ComplexF64}, d)) ≈ A rtol = tolerance
        @test_throws InexactError convert(PDMat{Float64}, d)
        @test Matrix(kron(d, PDMat(Hermitian(A[1:2, 1:2])))) ≈ kron(A, A[1:2, 1:2]) rtol = tolerance
        @test Matrix(d + PDiagMat(R[1, 2, 3])) ≈ A + Diagonal(R[1, 2, 3]) rtol = tolerance
        @test Matrix(ScalMat(3, R(2)) + d) ≈ A + 2I rtol = tolerance
        @test pdadd(Matrix(A), d) ≈ 2A rtol = tolerance
        @test pdadd(Matrix(A), d, R(2)) ≈ 3A rtol = tolerance
        r = copy(A)
        @test pdadd!(r, d) === r
        @test r ≈ 2A rtol = tolerance
        standard_deviations = sqrt.(real.(diag(A)))
        correlation = A ./ (standard_deviations * standard_deviations')
        @test Matrix(StatsBase.cov2cor(d, standard_deviations)) ≈ correlation rtol = tolerance
        @test Matrix(StatsBase.cor2cov(PDMat(correlation), standard_deviations)) ≈ A rtol = tolerance
        # A complex phase is not a positive-definite scaling operation.
        @test !(d * (1 + im) isa AbstractPDMat)
    end

    @testset "whitening and complex workspaces" begin
        for a in (PDMat(A), PDMat(Hermitian(A)), PDMat(A, lower), upper, lower)
            for (f, f!, expected) in (
                    (whiten, whiten!, L \ X),
                    (invwhiten, invwhiten!, U * X),
                    (unwhiten, unwhiten!, L * X),
                    (invunwhiten, invunwhiten!, U \ X),
                )
                @test f(a, X) ≈ expected rtol = tolerance
                r = similar(X)
                @test f!(r, a, X) === r
                @test r ≈ expected rtol = tolerance
                aliased = copy(X)
                @test f!(aliased, a, aliased) === aliased
                @test aliased ≈ expected rtol = tolerance
            end
            expected_quad = real.([dot(X[:, j], A * X[:, j]) for j in axes(X, 2)])
            expected_invquad = real.([dot(X[:, j], A \ X[:, j]) for j in axes(X, 2)])
            for (f, f!, expected) in ((quad, quad!, expected_quad), (invquad, invquad!, expected_invquad))
                measured = f(a, X)
                @test eltype(measured) === R
                @test measured ≈ expected rtol = tolerance
                @test all(>(0), measured)
                @test f(a, x) isa R
                @test f(a, x) ≈ expected[1] rtol = tolerance
                r = zeros(R, size(X, 2))
                @test f!(r, a, X) === r
                @test r ≈ expected rtol = tolerance
                @test f(a, @view X[:, 1:2]) ≈ expected rtol = tolerance
                @test isempty(f(a, zeros(T, 3, 0)))
            end
        end
        # Generic raw-matrix entry points and the internal dense Cholesky paths.
        @test whiten(A, X) ≈ L \ X rtol = tolerance
        @test unwhiten(A, X) ≈ L * X rtol = tolerance
        @test PDMats.chol_lower(A) ≈ L rtol = tolerance
        @test PDMats.chol_upper(A) ≈ U rtol = tolerance
        @test quad(A, X) ≈ real.([dot(X[:, j], A * X[:, j]) for j in axes(X, 2)]) rtol = tolerance
        # A real covariance with complex observations has the same workspace need.
        real_d = PDMat(Matrix{R}(I, 3, 3))
        @test quad(real_d, X) ≈ vec(sum(abs2, X; dims = 1)) rtol = tolerance
        @test invquad(real_d, X) ≈ vec(sum(abs2, X; dims = 1)) rtol = tolerance
    end

    @testset "adjoint congruences and unchanged real wrappers" begin
        B = X'
        for a in (PDMat(A), upper, lower)
            for (f, input, expected) in (
                    (X_A_Xt, B, B * A * B'),
                    (Xt_A_X, X, X' * A * X),
                    (X_invA_Xt, B, B * (A \ B')),
                    (Xt_invA_X, X, X' * (A \ X)),
                )
                measured = f(a, input)
                @test measured isa Hermitian
                @test ishermitian(measured)
                @test measured ≈ expected rtol = tolerance
            end
        end
        real_a = PDMat(Matrix{R}(I, 3, 3))
        real_x = real.(X)
        @test Xt_A_X(real_a, real_x) isa Symmetric
        @test X_A_Xt(real_a, real_x') isa Symmetric
        @test Xt_A_X(real_a, X) isa Hermitian
        @test Xt_A_X(PDMat(A), real_x) isa Hermitian
        @test Xt_A_X(cholesky(Matrix{R}(I, 3, 3)), real_x) isa Symmetric
    end

    @testset "invalid matrices and dimensions" begin
        nonhermitian = copy(A)
        nonhermitian[1, 2] += im
        @test_throws PosDefException PDMat(nonhermitian)
        indefinite = copy(A)
        indefinite[1, 1] = -1
        @test_throws PosDefException PDMat(indefinite)
        @test_throws DimensionMismatch PDMat(A, cholesky(Hermitian(A[1:2, 1:2])))
        @test_throws DimensionMismatch quad(PDMat(A), zeros(T, 2))
        @test_throws DimensionMismatch quad!(zeros(R, 1), PDMat(A), X)
        @test_throws DimensionMismatch Xt_A_X(PDMat(A), zeros(T, 2, 2))
    end
end
