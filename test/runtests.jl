include("testutils.jl")
tests = ["pdmtypes", "addition", "generics", "kron", "staticarrays"]
println("Running tests ...")

for t in tests
    println("* $t ")
    include("$t.jl")
end
