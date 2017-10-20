
# test operators with pd matrix types
using PDMats
using Compat.Test

# test scalar multiplication 
print_with_color(:blue, "Testing scalar multiplication\n")
pm1 = PDMat(eye(3))
pm2 = PDiagMat(ones(3))
pm3 = ScalMat(3,1)

pm1a = PDMat(3.0 .* eye(3))
pm2a = PDiagMat(3.0 .* ones(3))
pm3a = ScalMat(3, 3)

pmats = Any[pm1, pm2, pm3]
pmatsa= Any[pm1a,pm2a,pm3a]

for i in 1:length(pmats)
    @test full(3.0 * pmats[i]) == full(pmatsa[i])
    @test full(pmats[i] * 3.0) == full(pmatsa[i])
    @test full(3 * pmats[i])   == full(pmatsa[i])
    @test full(pmats[i] * 3)   == full(pmatsa[i])
end
