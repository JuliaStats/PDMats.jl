include("testutils.jl")
tests = ["pdmtypes", "abstracttypes", "addition", "congruence", "generics", "kron", "chol", "specialarrays", "sqrt", "ad", "ext/statsbase"]
println("Running tests ...")

for t in tests
    println("* $t ")
    include("$t.jl")
end
