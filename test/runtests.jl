include("testutils.jl")
tests = ["pdmtypes", "abstracttypes", "addition", "generics", "kron", "chol",
         "specialarrays", "sqrt", "ad"]
println("Running tests ...")

for t in tests
    println("* $t ")
    include("$t.jl")
end
