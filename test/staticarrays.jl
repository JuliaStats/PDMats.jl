using StaticArrays

@testset "StaticArrays" begin
    X = @SMatrix rand(10,4);
    S = @SMatrix(randn(4,7)) |> x -> Symmetric(x * x');
    PDS = PDMat(S)
    @test isbits(PDS)
    @test isbits(X / PDS)
    @test isbits(PDS \ X')
    @test (X / PDS) ≈ Matrix(X) / Matrix(S)
    @test (PDS \ X') ≈ Matrix(S) \ Matrix(X')

    D = PDiagMat(@SVector(rand(4)))
    @test isbits(D)
    @test isbits(X / D)
    @test isbits(D \ X')
    @test (X / D) ≈ Matrix(X) / Matrix(D)
    @test (D \ X') ≈ Matrix(D) \ Matrix(X')
end
