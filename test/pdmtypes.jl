using LinearAlgebra, PDMats, SparseArrays
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
            if HAVE_CHOLMOD
                s = SparseMatrixCSC{T}(I, 2, 2)
                @test PDMat(s, cholesky(s)).mat == PDMat(s).mat == PDMat(cholesky(s)).mat
            end
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
            if HAVE_CHOLMOD
                @testset "PDMat from sparse matrix" begin
                    test_pdmat(PDMat(sparse(M)), M, cmat_eq=true, verbose=1, t_eig=false)
                end
            end
        end

        @testset "test deprecated internal constructors" begin
            m = Matrix{T}(I, 2, 2)
            C = cholesky(m)
            @test @test_deprecated(PDMat{T,typeof(m)}(2, m, C)) == PDMat(m)
            d = ones(T,2)
            @test @test_deprecated(PDiagMat(2, d)) == @test_deprecated(PDiagMat{T,Vector{T}}(2, d)) == PDiagMat(d)
            if HAVE_CHOLMOD
                s = SparseMatrixCSC{T}(I, 2, 2)
                @test @test_deprecated(PDMat{T, typeof(s)}(2, s, cholesky(s))) == PDMat(s)
            end
        end
    end

    @testset "zero-dimensional matrices" begin
        Z = zeros(0, 0)
        test_pdmat(PDMat(Z), Z; t_eig=false)
        test_pdmat(PDiagMat(diag(Z)), Z; t_eig=false)
    end

    @testset "float type conversions" begin
        for T in (Float32, Float64), S in (Float32, Float64)
            A = PDMat(Matrix{T}(I, 2, 2))
            for R in (AbstractArray{S}, AbstractMatrix{S}, AbstractPDMat{S}, PDMat{S})
                B = @inferred(convert(R, A))
                @test B isa PDMat{S}
                @test B == A
                @test (B === A) === (S === T)
                @test (B.mat === A.mat) === (S === T)
                @test (B.fact === A.fact) === (S === T)
            end

            A = PDiagMat(ones(T, 2))
            for R in (AbstractArray{S}, AbstractMatrix{S}, AbstractPDMat{S}, PDiagMat{S})
                B = @inferred(convert(R, A))
                @test B isa PDiagMat{S}
                @test B == A
                @test (B === A) === (S === T)
                @test (B.diag === A.diag) === (S === T)
            end

            A = ScalMat(4, T(1))
            for R in (AbstractArray{S}, AbstractMatrix{S}, AbstractPDMat{S}, ScalMat{S})
                B = @inferred(convert(R, A))
                @test B isa ScalMat{S}
                @test B == A
                @test (B === A) === (S === T)
                @test (B.value === A.value) === (S === T)
            end

            if HAVE_CHOLMOD
                A = PDMat(SparseMatrixCSC{T}(I, 2, 2))
                for R in (AbstractArray{S}, AbstractMatrix{S}, AbstractPDMat{S}, PDMat{S})
                    B = @inferred(convert(R, A))
                    @test B isa PDMat{S}
                    @test B == A
                    @test (B === A) === (S === T)
                    @test (B.mat === A.mat) === (S === T)
                    # CholMOD only supports Float64 and ComplexF64 type parameters!
                    # Hence the Cholesky factorization is reused
                    @test B.fact === A.fact
                end
            end
        end
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

        for M in (PDiagMat(vec(A)), ScalMat(1, first(A)))
            z = x / M
            @test typeof(z) === typeof(y)
            @test size(z) == size(y)
            @test z ≈ y
        end

        # requires https://github.com/JuliaLang/julia/pull/32594
        if VERSION >= v"1.3.0-DEV.562"
            z = x / PDMat(A)
            @test typeof(z) === typeof(y)
            @test size(z) == size(y)
            @test z ≈ y
        end

        # right division not defined for CHOLMOD:
        # `rdiv!(::Matrix{Float64}, ::SparseArrays.CHOLMOD.Factor{Float64})` not defined
        if !HAVE_CHOLMOD
            z = x / PDMat(sparse(first(A), 1, 1)) 
            @test typeof(z) === typeof(y)
            @test size(z) == size(y)
            @test z ≈ y
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
        @test cholesky(M) isa Cholesky
        @test Matrix(M) ≈ A

        M = @inferred AbstractPDMat(cholesky(A))
        @test M isa PDMat
        @test cholesky(M) isa Cholesky
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
        @test M isa PDMat
        @test cholesky(M) isa CHOLMOD.Factor
        @test Matrix(M) ≈ A

        if VERSION < v"1.6"
            # inference fails e.g. on Julia 1.0
            M = AbstractPDMat(cholesky(sparse(A)))
        else
            M = @inferred AbstractPDMat(cholesky(sparse(A)))
        end
        @test M isa PDMat
        @test cholesky(M) isa CHOLMOD.Factor
        @test Matrix(M) ≈ A
    end

    @testset "properties and fields" begin
        for dim in (1, 5, 10)
            x = rand(dim, dim)
            M = PDMat(x' * x + I)
            @test fieldnames(typeof(M)) == (:mat, :chol)
            @test propertynames(M) == (fieldnames(typeof(M))..., :dim)
            @test getproperty(M, :dim) === dim
            for p in fieldnames(typeof(M))
                @test getproperty(M, p) === getfield(M, p)
            end

            M = PDiagMat(rand(dim))
            @test fieldnames(typeof(M)) == (:diag,)
            @test propertynames(M) == (fieldnames(typeof(M))..., :dim)
            @test getproperty(M, :dim) === dim
            for p in fieldnames(typeof(M))
                @test getproperty(M, p) === getfield(M, p)
            end

            M = ScalMat(dim, rand())
            @test fieldnames(typeof(M)) == (:dim, :value)
            @test propertynames(M) == fieldnames(typeof(M))
            for p in fieldnames(typeof(M))
                @test getproperty(M, p) === getfield(M, p)
            end

            if HAVE_CHOLMOD
                x = sprand(dim, dim, 0.2)
                M = PDMat(x' * x + I)
                @test fieldnames(typeof(M)) == (:mat, :chol)
                @test propertynames(M) == (fieldnames(typeof(M))..., :dim)
                @test getproperty(M, :dim) === dim
                for p in fieldnames(typeof(M))
                    @test getproperty(M, p) === getfield(M, p)
                end
            end
        end
    end

    @testset "Incorrect dimensions" begin
        x = rand(10, 10)
        A = x * x' + I
        C = cholesky(A)
        @test_throws DimensionMismatch PDMat(A[:, 1:(end - 1)], C)
        @test_throws DimensionMismatch PDMat(A[1:(end - 1), 1:(end - 1)], C)

        if HAVE_CHOLMOD
            x = sprand(10, 10, 0.2)
            A = x * x' + I
            C = cholesky(A)
            @test_throws DimensionMismatch PDMat(A[:, 1:(end - 1)], C)
            @test_throws DimensionMismatch PDMat(A[1:(end - 1), 1:(end - 1)], C)
        end
    end
end
