include("testutils.jl")
tests = ["pdmtypes", "addition", "generics", "kron"]
println("Running tests ...")

for t in tests
    println("* $t ")
    include("$t.jl")
end
