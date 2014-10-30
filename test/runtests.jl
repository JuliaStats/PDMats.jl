


tests = ["pdmtypes"]
println("Running tests ...")

for t in tests
	println("* $t ")
	include("$t.jl")
end


