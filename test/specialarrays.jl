using BandedMatrices
using StaticArrays

@testset "Special matrix types" begin
    @testset "StaticArrays" begin
        # Full matrix
        S = (x -> x * x' + I)(@SMatrix(randn(4, 7)))
        PDS = PDMat(S)
        @test PDS isa PDMat{Float64, <:SMatrix{4, 4, Float64}}
        @test isbits(PDS)
        C = cholesky(S)
        PDC = PDMat(C)
        @test typeof(PDC) === typeof(PDS)
        @test Matrix(PDC) ≈ Matrix(PDS)
        @test PDMat(S, C) === PDS
        @test @allocated(PDMat(S)) == @allocated(PDMat(C)) == @allocated(PDMat(S, C))

        # Diagonal matrix
        D = PDiagMat(@SVector(rand(4)))
        @test D isa PDiagMat{Float64, <:SVector{4, Float64}}
        @test @inferred(kron(D, D)) isa PDiagMat{Float64, <:SVector{16, Float64}}

        # Scaled identity matrix
        E = ScalMat(4, 1.2)

        x = @SVector rand(4)
        X = @SMatrix rand(10, 4)
        Y = @SMatrix rand(4, 10)

        for A in (PDS, D, E, C)
            if !(A isa Cholesky)
                # `*(::Cholesky, ::SArray)` is not defined
                @test A * x isa SVector{4, Float64}
                @test A * x ≈ Matrix(A) * Vector(x)

                @test A * Y isa SMatrix{4, 10, Float64}
                @test A * Y ≈ Matrix(A) * Matrix(Y)
            end

            @test X / A isa SMatrix{10, 4, Float64}
            @test X / A ≈ Matrix(X) / Matrix(A)

            @test A \ x isa SVector{4, Float64}
            @test A \ x ≈ Matrix(A) \ Vector(x)

            @test A \ Y isa SMatrix{4, 10, Float64}
            @test A \ Y ≈ Matrix(A) \ Matrix(Y)

            @test whiten(A, x) isa SVector{4, Float64}
            @test whiten(A, x) ≈ cholesky(Matrix(A)).L \ Vector(x)

            @test whiten(A, Y) isa SMatrix{4, 10, Float64}
            @test whiten(A, Y) ≈ cholesky(Matrix(A)).L \ Matrix(Y)

            @test unwhiten(A, x) isa SVector{4, Float64}
            @test unwhiten(A, x) ≈ cholesky(Matrix(A)).L * Vector(x)

            @test unwhiten(A, Y) isa SMatrix{4, 10, Float64}
            @test unwhiten(A, Y) ≈ cholesky(Matrix(A)).L * Matrix(Y)

            @test quad(A, x) isa Float64
            @test quad(A, x) ≈ Vector(x)' * Matrix(A) * Vector(x)

            @test quad(A, Y) isa SVector{10, Float64}
            @test quad(A, Y) ≈ diag(Matrix(Y)' * Matrix(A) * Matrix(Y))

            @test invquad(A, x) isa Float64
            @test invquad(A, x) ≈ Vector(x)' * (Matrix(A) \ Vector(x))

            @test invquad(A, Y) isa SVector{10, Float64}
            @test invquad(A, Y) ≈ diag(Matrix(Y)' * (Matrix(A) \ Matrix(Y)))

            @test X_A_Xt(A, X) isa SMatrix{10, 10, Float64}
            @test X_A_Xt(A, X) ≈ Matrix(X) * Matrix(A) *  Matrix(X)'

            @test X_invA_Xt(A, X) isa SMatrix{10, 10, Float64}
            @test X_invA_Xt(A, X) ≈ Matrix(X) * (Matrix(A) \ Matrix(X)')

            @test Xt_A_X(A, Y) isa SMatrix{10, 10, Float64}
            @test Xt_A_X(A, Y) ≈ Matrix(Y)' * Matrix(A) * Matrix(Y)

            @test Xt_invA_X(A, Y) isa SMatrix{10, 10, Float64}
            @test Xt_invA_X(A, Y) ≈ Matrix(Y)' * (Matrix(A) \ Matrix(Y))
        end
    end

    @testset "BandedMatrices" begin
        # Full matrix
        A = Symmetric(BandedMatrix(Eye(5), (1, 1)))
        P = PDMat(A)
        @test P isa PDMat{Float64, <:BandedMatrix{Float64}}

        x = rand(5)
        X = rand(2, 5)
        Y = rand(5, 2)
        @test P * x ≈ x
        @test P * Y ≈ Y
        @test X / P ≈ X
        @test P \ x ≈ x
        @test P \ Y ≈ Y
        @test X_A_Xt(P, X) ≈ X * X'
        @test X_invA_Xt(P, X) ≈ X * X'
        @test Xt_A_X(P, Y) ≈ Y' * Y
        @test Xt_invA_X(P, Y) ≈ Y' * Y
    end
end
