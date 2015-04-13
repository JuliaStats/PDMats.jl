# addition of positive definite matrices

using PDMats
using Base.Test

C1 = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
va = [1.5, 2.5, 2.0]

pm1 = PDMat(C1)
pm2 = PDiagMat(va)
pm3 = ScalMat(3, 2.0)
pm4 = 2.0I

pmats = Any[pm1, pm2, pm3]

for p1 in pmats, p2 in pmats
	pr = p1 + p2
	@test size(pr) == size(p1)
	@test_approx_eq full(pr) full(p1) + full(p2)

	pr = pdadd(p1, p2, 1.5)
	@test size(pr) == size(p1)
	@test_approx_eq full(pr) full(p1) + full(p2) * 1.5
end

for p1 in pmats
        pr = p1 + pm4
        @test size(pr) == size(p1)
        @test_approx_eq full(pr) full(p1) + pm4
end
