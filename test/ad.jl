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
            a -> f!(Vector{promote_type(eltype(a), Float64)}(undef, 4),
                    PDMat(Symmetric(reshape(a, 4, 4))), ones(4))
        end

        J = only(FiniteDifferences.jacobian(fdm, apply_f, a))
        @test only(FiniteDifferences.jacobian(fdm, apply_f!, a)) ≈ J

        @test ForwardDiff.jacobian(apply_f, a) ≈ J
        @test ForwardDiff.jacobian(apply_f!, a) ≈ J
    end
end
