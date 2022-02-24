include("testutils.jl")
tests = ["pdmtypes", "addition", "generics", "kron", "chol", "specialarrays"]
println("Running tests ...")

for t in tests
    println("* $t ")
    include("$t.jl")
end
