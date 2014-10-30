
# between pdmat and pdmat

+ (a::PDMat, b::AbstractPDMat) = PDMat(a.mat + full(b))
+ (a::PDiagMat, b::AbstractPDMat) = PDMat(_adddiag!(full(b), a.diag))
+ (a::ScalMat, b::AbstractPDMat) = PDMat(_adddiag!(full(b), a.value))

+ (a::PDMat, b::PDMat) = PDMat(a.mat + b.mat)
+ (a::PDMat, b::PDiagMat) = PDMat(_adddiag(a.mat, b.diag))
+ (a::PDMat, b::ScalMat) = PDMat(_adddiag(a.mat, b.value))

+ (a::PDiagMat, b::PDMat) = PDMat(_adddiag(b.mat, a.diag))
+ (a::PDiagMat, b::PDiagMat) = PDiagMat(a.diag + b.diag)
+ (a::PDiagMat, b::ScalMat) = PDiagMat(a.diag .+ b.value)

+ (a::ScalMat, b::PDMat) = PDMat(_adddiag(b.mat, a.value))
+ (a::ScalMat, b::PDiagMat) = PDiagMat(a.value .+ b.diag)
+ (a::ScalMat, b::ScalMat) = ScalMat(a.dim, a.value + b.value)

pdadd(a::PDMat, b::AbstractPDMat, c::Float64) = PDMat(a.mat + full(b * c))
pdadd(a::PDiagMat, b::AbstractPDMat, c::Float64) = PDMat(_adddiag!(full(b * c), a.diag, 1.0))
pdadd(a::ScalMat, b::AbstractPDMat, c::Float64) = PDMat(_adddiag!(full(b * c), a.value))

pdadd(a::PDMat, b::PDMat, c::Float64) = PDMat(a.mat + b.mat * c)
pdadd(a::PDMat, b::PDiagMat, c::Float64) = PDMat(_adddiag(a.mat, b.diag, c))
pdadd(a::PDMat, b::ScalMat, c::Float64) = PDMat(_adddiag(a.mat, b.value * c))

pdadd(a::PDiagMat, b::PDMat, c::Float64) = PDMat(_adddiag!(b.mat * c, a.diag, 1.0))
pdadd(a::PDiagMat, b::PDiagMat, c::Float64) = PDiagMat(a.diag + b.diag * c)
pdadd(a::PDiagMat, b::ScalMat, c::Float64) = PDiagMat(a.diag .+ b.value * c)

pdadd(a::ScalMat, b::PDMat, c::Float64) = PDMat(_adddiag!(b.mat * c, a.value))
pdadd(a::ScalMat, b::PDiagMat, c::Float64) = PDiagMat(a.value .+ b.diag * c)
pdadd(a::ScalMat, b::ScalMat, c::Float64) = ScalMat(a.dim, a.value + b.value * c)
