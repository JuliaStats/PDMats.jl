using LinearAlgebra, PDMats, Test

@testset "Congruency transforms with PD return values" begin
    shared_typemaps = [
        (ScalMat, ScalMat) => ScalMat,
        (ScalMat, PDiagMat) => PDiagMat,
        (PDiagMat, ScalMat) => PDiagMat,
        (PDiagMat, PDiagMat) => PDiagMat,
    ]
    typemaps = [shared_typemaps..., (PDMat, ScalMat) => PDMat, (PDMat, PDiagMat) => PDMat]
    inv_typemaps = shared_typemaps

    for (fs, tmaps) in
        [((X_A_Xt, Xt_A_X), typemaps), ((X_invA_Xt, Xt_invA_X), inv_typemaps)]
        @testset "$f(::$TA, ::$TB) -> $Tret" for ((TA, TB), Tret) in tmaps, f in fs
            @testset for T in [Float32, Float64],
                    n in [3, 5],
                    uplo in (TA <: PDMat ? ('L', 'U') : (nothing,))

                A = _rand(TA, T, n, uplo)
                B = _rand(TB, T, n, uplo)
                ret = @inferred f(A, B)
                @test ret isa Tret{T}
                @test ret ≈ f(A, Matrix(B))
                if Tret <: PDMat
                    @test cholesky(ret).uplo == cholesky(A).uplo
                    @test Matrix(cholesky(ret)) ≈ ret.mat
                end
            end
        end
    end

    # A sparse `PDMat` cannot rescale its `CHOLMOD.Factor`, so it rescales the matrix and
    # factorizes anew. The result must still be positive definite and stay sparse.
    @testset "$f(::PDSparseMat, ::$TB)" for f in (X_A_Xt, Xt_A_X), TB in (ScalMat, PDiagMat)
        @testset for T in (Float32, Float64), n in (3, 5)
            A = _randPDSparseMat(T, n)
            B = _rand(TB, T, n, nothing)
            ret = f(A, B)
            @test ret isa PDSparseMat{T}
            @test issparse(ret.mat)
            @test ret ≈ f(A, Matrix(B))
            @test_throws DimensionMismatch f(A, _rand(TB, T, n + 1, nothing))
        end
    end
end
