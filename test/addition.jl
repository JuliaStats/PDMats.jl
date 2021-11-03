# addition of positive definite matrices

using PDMats

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
end
