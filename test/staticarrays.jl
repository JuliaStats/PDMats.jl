using StaticArrays

@testset "StaticArrays" begin
    # Full matrix
    S = (x -> x * x')(@SMatrix(randn(4, 7)))
    PDS = PDMat(S)
    @test PDS isa PDMat{Float64, <:SMatrix{4, 4, Float64}}
    @test isbits(PDS)

    # Diagonal matrix
    D = PDiagMat(@SVector(rand(4)))
    @test D isa PDiagMat{Float64, <:SVector{4, Float64}}

    x = @SVector rand(4)
    X = @SMatrix rand(10, 4)
    Y = @SMatrix rand(4, 10)

    for A in (PDS, D)
        @test A * x isa SVector{4, Float64}
        @test A * x ≈ Matrix(A) * Vector(x)

        @test A * Y isa SMatrix{4, 10, Float64}
        @test A * Y ≈ Matrix(A) * Matrix(Y)

        @test X / A isa SMatrix{10, 4, Float64}
        @test X / A ≈ Matrix(X) / Matrix(A)

        @test A \ x isa SVector{4, Float64}
        @test A \ x ≈ Matrix(A) \ Vector(x)

        @test A \ Y isa SMatrix{4, 10, Float64}
        @test A \ Y ≈ Matrix(A) \ Matrix(Y)

        @test X_A_Xt(A, X) isa SMatrix{10, 10, Float64}
        @test X_A_Xt(A, X) ≈ X_A_Xt(PDMat(Matrix(A)), Matrix(X))

        @test X_invA_Xt(A, X) isa SMatrix{10, 10, Float64}
        @test X_invA_Xt(A, X) ≈ X_invA_Xt(PDMat(Matrix(A)), Matrix(X))

        @test Xt_A_X(A, Y) isa SMatrix{10, 10, Float64}
        @test Xt_A_X(A, Y) ≈ Xt_A_X(PDMat(Matrix(A)), Matrix(Y))

        @test Xt_invA_X(A, Y) isa SMatrix{10, 10, Float64}
        @test Xt_invA_X(A, Y) ≈ Xt_invA_X(PDMat(Matrix(A)), Matrix(Y))
    end
end
