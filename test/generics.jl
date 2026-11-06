# test operators with pd matrix types
using LinearAlgebra, SparseArrays, PDMats
using Test

@testset "vector quadratic triple products" for T in (Float32, Float64)
    L = T[2 0 0; 1 3 0; -1 0.5 2]
    matrices = (PDMat(L * L'), PDiagMat(T[1, 3, 5]), ScalMat(3, T(2.5)))
    x, y = T[0.25, -0.5, 1.25], T[-1, 2, 0.5]
    saved_x = copy(x)
    for a in matrices, wrap in (adjoint, transpose)
        dense = Matrix(a)
        @test (@inferred *(wrap(x), a, x)) ≈ dot(x, dense * x)
        @test typeof(wrap(x) * a * x) === typeof(quad(a, x))

        # Equal values in a separate vector still use the bilinear operation.
        separate = copy(x)
        @test separate !== x
        @test wrap(x) * a * separate ≈ (wrap(x) * dense) * separate
        @test wrap(x) * a * y ≈ (wrap(x) * dense) * y
        mixed = Float64.(y)
        @test wrap(x) * a * mixed ≈ (wrap(x) * dense) * mixed

        storage = T[0.25, 99, -0.5, 99, 1.25]
        strided = @view storage[1:2:5]
        @test wrap(strided) * a * strided ≈ dot(x, dense * x)
        @test x == saved_x
        @test_throws DimensionMismatch wrap(x) * a * T[1, 2]
        @test_throws DimensionMismatch wrap(T[1, 2]) * a * T[1, 2]
        @test_throws DimensionMismatch wrap(T[1, 2, 3, 4]) * a * x

        # Reordering a non-finite expression can change Inf/NaN propagation.
        for value in (T(Inf), T(-Inf), T(NaN))
            nonfinite = T[value, 0, 1]
            @test isequal(wrap(nonfinite) * a * nonfinite, (wrap(nonfinite) * a) * nonfinite)
        end
    end
    empty_vector = T[]
    for a in (PDMat(Matrix{T}(I, 0, 0)), PDiagMat(T[]), ScalMat(0, one(T))), wrap in (adjoint, transpose)
        @test iszero(wrap(empty_vector) * a * empty_vector)
        @test typeof(wrap(empty_vector) * a * empty_vector) === typeof(quad(a, empty_vector))
    end
end

@testset "weighted product order and numeric promotion" begin
    for (T, exponent) in ((Float32, 80), (Float64, 600)), wrap in (adjoint, transpose)
        for sign in (-one(T), one(T)), power in (-exponent, exponent)
            x = T[sign * T(2)^power]
            weight = T(2)^(-power)
            expected = T(2)^power
            for a in (PDiagMat(T[weight]), ScalMat(1, weight))
                reference = (wrap(x) * a) * x
                @test reference === expected
                @test (@inferred *(wrap(x), a, x)) === reference
                @test wrap(x) * a * x === reference
            end
        end
    end
    for (x, weight) in (
                (Int8[7], Int8(5)), (Int8[7], Int16(5)), (Int16[200], Int8(5)),
                (Int8[7], Float32(0.5)), (Float32[7], Int8(5)),
                (Float64[7], Float32(0.5)), (Float32[7], Float64(0.5)),
                (Rational{Int}[3 // 2], 2 // 3), (BigInt[7], big(5)),
            ), wrap in (adjoint, transpose), a in (PDiagMat([weight]), ScalMat(1, weight))
        reference = (wrap(x) * a) * x
        result = wrap(x) * a * x
        @test typeof(result) === typeof(reference)
        @test result == reference
    end
end

@testset "small contributions survive large quadratic terms" begin
    small_count = 32768
    for (T, exponent) in ((Float32, 12), (Float64, 27))
        big = T(2)^exponent
        expected = big * big + T(small_count)
        large_first = vcat(big, ones(T, small_count))
        for x in (large_first, reverse(large_first)), wrap in (adjoint, transpose),
                a in (PDiagMat(ones(T, length(x))), ScalMat(length(x), one(T)))
            @test (@inferred *(wrap(x), a, x)) === expected
        end
    end
end

@testset "exceptional products retain the original BLAS path" begin
    for (T, powers) in ((Float32, (-70, -80, 80)), (Float64, (-530, -600, 600))),
            power in powers, wrap in (adjoint, transpose),
            a in (PDiagMat(T[1]), ScalMat(1, one(T)))
        x = T[T(2)^power]
        @test isequal(wrap(x) * a * x, (wrap(x) * a) * x)
        @test typeof(wrap(x) * a * x) === typeof((wrap(x) * a) * x)
    end
end

@testset "scalar multiplication" begin
    printstyled("Testing scalar multiplication\n"; color = :blue)
    pm1 = PDMat(Matrix(1.0I, 3, 3))
    pm2 = PDiagMat(ones(3))
    pm3 = ScalMat(3, 1)

    pm1a = PDMat(Matrix(3.0I, 3, 3))
    pm2a = PDiagMat(3.0 .* ones(3))
    pm3a = ScalMat(3, 3)

    pmats = Any[pm1, pm2, pm3]
    pmatsa = Any[pm1a, pm2a, pm3a]

    for i in 1:length(pmats)
        @test Matrix(3.0 * pmats[i]) == Matrix(pmatsa[i])
        @test Matrix(pmats[i] * 3.0) == Matrix(pmatsa[i])
        @test Matrix(3 * pmats[i]) == Matrix(pmatsa[i])
        @test Matrix(pmats[i] * 3) == Matrix(pmatsa[i])
    end
end

# issue #121
@test isposdef(PDMat([1.0 0.0; 0.0 1.0]))
@test isposdef(PDiagMat([1.0, 1.0]))
@test isposdef(ScalMat(2, 3.0))

@testset "ldiv!(A, b)" begin
    printstyled("Testing ldiv!(A, b)\n"; color = :blue)
    for A in (PDMat([4.0 2.0; 2.0 3.0]), PDiagMat([4.0, 3.0]), ScalMat(2, 4.0) #=PDSparseMat(sparse([4.0 2.0; 2.0 3.0]))=#) # CHOLMOD.Factor is not supported by ldiv!
        b = [1.0, 2.0]
        x = copy(b)
        ldiv!(A, x)
        @test isapprox(A * x, b)
        B = [1.0 2.0; 3.0 4.0]
        X = copy(B)
        ldiv!(A, X)
        @test isapprox(A * X, B)
    end
end
