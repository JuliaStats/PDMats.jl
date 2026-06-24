using PDMats, LinearAlgebra, Test

# Custom subtype for the scalar-multiplication test below; must be top-level since a
# `struct` cannot be defined inside a `@testset`.
struct MyPD <: AbstractPDMat{Float64}
    value::Float64
end
Base.size(::MyPD) = (2, 2)
Base.getindex(a::MyPD, i::Int, j::Int) = i == j ? a.value : 0.0
Base.:*(a::MyPD, c::Real) = MyPD(a.value * c)

@testset "type parameters" begin
    @testset "PDMat quad/invquad specialization for Matrix arguments" begin
        n, k = 200, 500
        M = randn(n, n)
        a = PDMat(M * transpose(M) + n * I)
        x = randn(n, k)
        xcol = x[:, 1]

        # These also compile the Matrix-argument methods used in the allocation checks below.
        @test quad(a, x) ≈ diag(transpose(x) * Matrix(a) * x)
        @test invquad(a, x) ≈ diag(transpose(x) * (a \ x))

        # The specialization reuses one length-n buffer across all columns, so the k-column
        # call allocates the k-element result plus no more scratch than a single column does.
        # The generic fallback instead forms the whole n×k product (~140x more here).
        overhead = 1024
        quad(a, xcol); invquad(a, xcol)
        @test (@allocated quad(a, x)) ≤ sizeof(quad(a, x)) + (@allocated quad(a, xcol)) + overhead
        @test (@allocated invquad(a, x)) ≤ sizeof(invquad(a, x)) + (@allocated invquad(a, xcol)) + overhead

        @test a isa PDMat{<:Real, <:Matrix}
        @test !(a isa PDMat{<:Real, <:Vector})

        for T in (Float64, Float32)
            b = PDMat(Matrix{T}(M * transpose(M) + n * I))
            y = randn(T, n, k)
            @inferred quad(b, y)
            @inferred invquad(b, y)
        end
    end

    @testset "scalar multiplication via generic fallbacks" begin
        # `c * a` and `a / c` delegate to a type's own `a * c` without recursing.
        a = MyPD(3.0)
        @test (a * 2.0).value == 6.0
        @test (2.0 * a).value == 6.0
        @test (a / 2.0).value == 1.5
    end

    @testset "division accepts mismatched operand element types" begin
        n = 4
        mats = Any[_randPDMat(Float64, n), _randPDiagMat(Float64, n), _randScalMat(Float64, n)]
        HAVE_CHOLMOD && push!(mats, _randPDSparseMat(Float64, n))
        operands = (rand(1:5, n), rand(1:5, n, 2), randn(Float32, n), randn(Float32, n, 2))
        for a in mats
            M = Matrix(a)
            for x in operands
                @test a \ x ≈ M \ Array(x)
                if !(a isa PDSparseMat)  # CHOLMOD has no right division
                    xt = permutedims(x)
                    @test xt / a ≈ Array(xt) / M
                end
            end
        end
    end

    @testset "ScalMat typed conversion" begin
        a = ScalMat(3, 2.0)
        @test Matrix(a) isa Matrix{Float64}
        @test Matrix(a) == [2.0 0 0; 0 2 0; 0 0 2]
        @test Matrix{Float32}(a) isa Matrix{Float32}
        @test Matrix{Float32}(a) == Float32[2 0 0; 0 2 0; 0 0 2]
    end
end
