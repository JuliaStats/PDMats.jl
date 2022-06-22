using LinearAlgebra, PDMats
using PDMats: chol_lower, chol_upper

@testset "chol_lower" begin
    d = 100
    A = rand(d, d)
    C = A'A
    invC = inv(C)
    size_of_one_copy = sizeof(C)
    @assert size_of_one_copy > d  # ensure the matrix is large enough that few-byte allocations don't matter

    chol_lower(C)
    @test (@allocated chol_lower(C)) < 1.05 * size_of_one_copy  # allow 5% overhead

    X = randn(d, 10)
    for uplo in (:L, :U)
        ch = cholesky(Symmetric(C, uplo))
        chol_lower(ch)
        @test (@allocated chol_lower(ch)) < 33  # allow small overhead for wrapper types
        chol_upper(ch)
        @test (@allocated chol_upper(ch)) < 33  # allow small overhead for wrapper types

        # Only test dim, `quad`/`invquad`, `whiten`/`unwhiten`, and tri products
        @test dim(ch) == size(C, 1)
        pdtest_quad(ch, C, invC, X, 0)
        pdtest_triprod(ch, C, invC, X, 0)
        pdtest_whiten(ch, C, 0)
    end
end
