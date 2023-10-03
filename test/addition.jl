# addition of positive definite matrices

using PDMats


struct ScalMat2D{T<:Real} <: AbstractPDMat{T}
    value::T
end

Base.Matrix(a::ScalMat2D) = Matrix(Diagonal(fill(a.value, 2)))

@testset "addition" begin
    for T in (Float64, Float32)
        printstyled("Testing addition with eltype = $T\n"; color=:blue)
        M = convert(Array{T,2}, [4.0 -2.0 -1.0; -2.0 5.0 -1.0; -1.0 -1.0 6.0])
        V = convert(Array{T,1}, [1.5, 2.5, 2.0])
        local X = convert(T, 2.0)

        pm1 = PDMat(M)
        pm2 = PDiagMat(V)
        pm3 = ScalMat(3, X)
        pm4 = X * I
        pm5 = PDSparseMat(sparse(M))

        pmats = Any[pm1, pm2, pm3] #, pm5]

        for p1 in pmats, p2 in pmats
            pr = p1 + p2
            @test size(pr) == size(p1)
            @test Matrix(pr) ≈ Matrix(p1) + Matrix(p2)

            pr = pdadd(p1, p2, convert(T, 1.5))
            @test size(pr) == size(p1)
            @test Matrix(pr) ≈ Matrix(p1) + Matrix(p2) * convert(T, 1.5)
        end

        for p1 in pmats
            pr = p1 + pm4
            @test size(pr) == size(p1)
            @test Matrix(pr) ≈ Matrix(p1) + pm4
        end
    end
    @testset "Abstract + Diag" for a in [PDiagMat([1,2]), ScalMat(2,1), PDiagMat(sparsevec([1.,0]))]
        M = ScalMat2D(1)
        @test Matrix(a + M) == Matrix(a) + Matrix(M)
    end
    @testset "PDMat + PDiagMat(sparse)" begin
        A = randn(2, 2)
        M = PDMat(A * A')
        Dsp = PDiagMat(sparsevec([1.0, 0]))
        @test Matrix(M + Dsp) == Matrix(M) + Matrix(Dsp)
    end
end
