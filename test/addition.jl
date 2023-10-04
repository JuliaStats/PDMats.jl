# addition of positive definite matrices

using PDMats


# New AbstractPDMat type for the tests below
# Supports only functions needed in the tests below
struct ScalMat3D{T<:Real} <: AbstractPDMat{T}
    value::T
end
Base.Matrix(a::ScalMat3D) = Matrix(Diagonal(fill(a.value, 3)))
Base.size(::ScalMat3D) = (3, 3)
# Not generally correct
Base.:*(a::ScalMat3D, c::Real) = ScalMat3D(a.value * c)
Base.getindex(a::ScalMat3D, i::Int, j::Int) = i == j ? a.value : zero(a.value)

@testset "addition" begin
    for T in (Float64, Float32)
        printstyled("Testing addition with eltype = $T\n"; color=:blue)
        M = convert(Array{T,2}, [4.0 -2.0 -1.0; -2.0 5.0 -1.0; -1.0 -1.0 6.0])
        V = convert(Array{T,1}, [1.5, 2.5, 2.0])
        local X = convert(T, 2.0)

        pm1 = PDMat(M)
        pm2 = PDiagMat(V)
        pm3 = PDiagMat(sparse(V))
        pm4 = ScalMat(3, X)
        pm5 = PDSparseMat(sparse(M))
        pm6 = ScalMat3D(X)

        pmats = Any[pm1, pm2, pm3, pm4, pm5, pm6]

        for p1 in pmats, p2 in pmats
            pr = p1 + p2
            @test size(pr) == size(p1)
            @test Matrix(pr) ≈ Matrix(p1) + Matrix(p2)

            if p1 isa ScalMat3D
                @test_broken pdadd(p1, p2, convert(T, 1.5))
            else
                pr = pdadd(p1, p2, convert(T, 1.5))
                @test size(pr) == size(p1)
                @test Matrix(pr) ≈ Matrix(p1) + Matrix(p2) * convert(T, 1.5)
            end
        end

        for p1 in pmats
            if p1 isa ScalMat3D
                @test_broken p1 + X * I
            else
                pr = p1 + X * I
                @test size(pr) == size(p1)
                @test Matrix(pr) ≈ Matrix(p1) + X * I
            end
        end
    end
end
