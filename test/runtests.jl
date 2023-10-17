include("testutils.jl")
tests = ["pdmtypes", "abstracttypes", "addition", "generics", "kron", "chol", "specialarrays", "sqrt", "ext"]
println("Running tests ...")

for t in tests
    println("* $t ")
    include("$t.jl")
end
