using PDMats
using FiniteDifferences
using ForwardDiff

using LinearAlgebra
using Test

# issue #217
@testset "PDMat: (un)whiten" begin
    a = vec(Matrix{Float64}(I, 4, 4))
    fdm = central_fdm(5, 1)

    for (f, f!) in ((whiten, whiten!), (unwhiten, unwhiten!))
        apply_f = let f = f
            a -> f(PDMat(Symmetric(reshape(a, 4, 4))), ones(4))
        end
        apply_f! = let f! = f!
            a -> f!(Vector{promote_type(eltype(a), Float64)}(undef, 4), PDMat(Symmetric(reshape(a, 4, 4))), ones(4))
        end

        J = only(FiniteDifferences.jacobian(fdm, apply_f, a))
        @test only(FiniteDifferences.jacobian(fdm, apply_f!, a)) ≈ J

        @test ForwardDiff.jacobian(apply_f, a) ≈ J
        @test ForwardDiff.jacobian(apply_f!, a) ≈ J
    end
end

# issue #99: `invquad`/`quad` must differentiate correctly and allocate only
# O(n·#partials) — not O(n²·#partials), as when the factor was promoted to `Dual`.
@testset "PDMat: quad/invquad with ForwardDiff (issue #99)" begin
    n = 100
    M = randn(n, n)
    A = PDMat(Symmetric(M * M' + n * I))
    x = randn(n)

    # ∇ invquad(A, x) = 2 A⁻¹x, ∇ quad(A, x) = 2 A x
    for (f, grad) in ((invquad, 2 * (A \ x)), (quad, 2 * (A * x)))
        @test ForwardDiff.gradient(Base.Fix1(f, A), x) ≈ grad
    end

    # only a single result vector the size of `xd` is allocated, not an O(n²) matrix
    xd = [ForwardDiff.Dual(xi, Tuple(randn(4))) for xi in x]
    for f in (invquad, quad)
        g = Base.Fix1(f, A)
        g(xd)  # warm up
        @test (@allocated g(xd)) ≤ 3 * sizeof(xd)
    end
end
