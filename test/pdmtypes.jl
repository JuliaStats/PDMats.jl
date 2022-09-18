using LinearAlgebra, PDMats, SparseArrays, SuiteSparse
using Test

@testset "pd matrix types" begin
    for T in [Float64, Float32]
        @testset "test that all external constructors are accessible" begin
            m = Matrix{T}(I, 2, 2)
            @test PDMat(m, cholesky(m)).mat == PDMat(Symmetric(m)).mat == PDMat(m).mat == PDMat(cholesky(m)).mat
            d = ones(T,2)
            @test @test_deprecated(PDiagMat(d, d)) == PDiagMat(d)
            x = one(T)
            @test @test_deprecated(ScalMat(2, x, x)) == ScalMat(2, x)
            s = SparseMatrixCSC{T}(I, 2, 2)
            @test PDSparseMat(s, cholesky(s)).mat == PDSparseMat(s).mat == PDSparseMat(cholesky(s)).mat
        end

        @testset "test the functionality" begin
            M = convert(Array{T,2}, [4. -2. -1.; -2. 5. -1.; -1. -1. 6.])
            V = convert(Array{T,1}, [1.5, 2.5, 2.0])
            X = convert(T,2.0)

            @testset "PDMat from Matrix" begin
                test_pdmat(PDMat(M), M,                        cmat_eq=true, verbose=1)
            end
            @testset "PDMat from Cholesky" begin
                cholL = Cholesky(Matrix(transpose(cholesky(M).factors)), 'L', 0)
                test_pdmat(PDMat(cholL), M,                    cmat_eq=true, verbose=1)
            end
            @testset "PDiagMat" begin
                test_pdmat(PDiagMat(V), Matrix(Diagonal(V)),   cmat_eq=true, verbose=1)
            end
            @testset "ScalMat" begin
                test_pdmat(ScalMat(3,X), X*Matrix{T}(I, 3, 3), cmat_eq=true, verbose=1)
            end
            @testset "PDSparseMat" begin
                test_pdmat(PDSparseMat(sparse(M)), M,          cmat_eq=true, verbose=1, t_eig=false)
            end
        end
    end

    @testset "zero-dimensional matrices" begin
        Z = zeros(0, 0)
        test_pdmat(PDMat(Z), Z; t_eig=false)
        test_pdmat(PDiagMat(diag(Z)), Z; t_eig=false)
    end

    @testset "float type conversions" begin
        m = Matrix{Float32}(I, 2, 2)
        @test convert(PDMat{Float64}, PDMat(m)).mat == PDMat(convert(Array{Float64}, m)).mat
        @test convert(AbstractArray{Float64}, PDMat(m)).mat == PDMat(convert(Array{Float64}, m)).mat
        m = ones(Float32,2)
        @test convert(PDiagMat{Float64}, PDiagMat(m)).diag == PDiagMat(convert(Array{Float64}, m)).diag
        @test convert(AbstractArray{Float64}, PDiagMat(m)).diag == PDiagMat(convert(Array{Float64}, m)).diag
        x = one(Float32); d = 4
        @test convert(ScalMat{Float64}, ScalMat(d, x)).value == ScalMat(d, convert(Float64, x)).value
        @test convert(AbstractArray{Float64}, ScalMat(d, x)).value == ScalMat(d, convert(Float64, x)).value
        s = SparseMatrixCSC{Float32}(I, 2, 2)
        @test convert(PDSparseMat{Float64}, PDSparseMat(s)).mat == PDSparseMat(convert(SparseMatrixCSC{Float64}, s)).mat
    end

    @testset "no-op conversion with correct eltype (#101)" begin
        X = PDMat((Y->Y'Y)(randn(Float32, 4, 4)))
        @test convert(AbstractArray{Float32}, X) === X
        @test convert(AbstractArray{Float64}, X) !== X
    end

    @testset "type stability of whiten! and unwhiten!" begin
        a = PDMat([1 0.5; 0.5 1])
        @inferred whiten!(ones(2), a, ones(2))
        @inferred unwhiten!(ones(2), a, ones(2))
        @inferred whiten(a, ones(2))
        @inferred unwhiten(a, ones(2))
    end

    @testset "convert Matrix type to the same Cholesky type (#117)" begin
        @test PDMat([1 0; 0 1]) == [1.0 0.0; 0.0 1.0]
    end

    # https://github.com/JuliaStats/PDMats.jl/pull/141
    @testset "PDiagMat with range" begin
        v = 0.1:0.1:0.5
        d = PDiagMat(v)
        @test d isa PDiagMat{eltype(v),typeof(v)}
        @test d.diag === v
    end

    @testset "division of vectors (size (1, 1))" begin
        A = rand(1, 1)
        x = randn(1)
        y = x / A
        @assert x / A isa Matrix{Float64}
        @assert size(y) == (1, 1)

        for M in (PDiagMat(vec(A)), ScalMat(1, first(A)))
            @test x / M isa Matrix{Float64}
            @test x / M ≈ y
        end

        # requires https://github.com/JuliaLang/julia/pull/32594
        if VERSION >= v"1.3.0-DEV.562"
            @test x / PDMat(A) isa Matrix{Float64}
            @test x / PDMat(A) ≈ y
        end

        # right division not defined for CHOLMOD:
        # `rdiv!(::Matrix{Float64}, ::SuiteSparse.CHOLMOD.Factor{Float64})` not defined
        if !HAVE_CHOLMOD
            @test x / PDSparseMat(sparse(first(A), 1, 1)) isa Matrix{Float64}
            @test x / PDSparseMat(sparse(first(A), 1, 1)) ≈ y
        end
    end

    @testset "PDMat from Cholesky decomposition of diagonal matrix (#137)" begin
        # U'*U where U isa UpperTriangular etc.
        # requires https://github.com/JuliaLang/julia/pull/33334
        if VERSION >= v"1.4.0-DEV.286"
            x = rand(10, 10)
            A = Diagonal(x' * x)
            M = PDMat(cholesky(A))
            @test M isa PDMat{Float64, typeof(A)}
            @test Matrix(M) ≈ A
        end
    end

    @testset "AbstractPDMat constructors (#136)" begin
        x = rand(10, 10)
        A = x' * x + I

        M = @inferred AbstractPDMat(A)
        @test M isa PDMat
        @test Matrix(M) ≈ A

        M = @inferred AbstractPDMat(cholesky(A))
        @test M isa PDMat
        @test Matrix(M) ≈ A

        M = @inferred AbstractPDMat(Diagonal(A))
        @test M isa PDiagMat
        @test Matrix(M) ≈ Diagonal(A)

        M = @inferred AbstractPDMat(Symmetric(Diagonal(A)))
        @test M isa PDiagMat
        @test Matrix(M) ≈ Diagonal(A)

        M = @inferred AbstractPDMat(Hermitian(Diagonal(A)))
        @test M isa PDiagMat
        @test Matrix(M) ≈ Diagonal(A)

        M = @inferred AbstractPDMat(sparse(A))
        @test M isa PDSparseMat
        @test Matrix(M) ≈ A

        if VERSION < v"1.6"
            # inference fails e.g. on Julia 1.0
            M = AbstractPDMat(cholesky(sparse(A)))
        else
            M = @inferred AbstractPDMat(cholesky(sparse(A)))
        end
        @test M isa PDSparseMat
        @test Matrix(M) ≈ A
    end
end
