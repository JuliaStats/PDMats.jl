using LinearAlgebra, PDMats
using PDMats: chol_lower, chol_upper

@testset "chol_lower" begin
    A = rand(100, 100)
    C = A'A
    size_of_one_copy = sizeof(C)
    @assert size_of_one_copy > 100  # ensure the matrix is large enough that few-byte allocations don't matter

    chol_lower(C)
    @test (@allocated chol_lower(C)) < 1.05 * size_of_one_copy  # allow 5% overhead

    for uplo in (:L, :U)
        ch = cholesky(Symmetric(C, uplo))
        chol_lower(ch)
        @test (@allocated chol_lower(ch)) < 33  # allow small overhead for wrapper types
        chol_upper(ch)
        @test (@allocated chol_upper(ch)) < 33  # allow small overhead for wrapper types
    end
end
