
# between pdmat and pdmat

+(a::PDMat, b::AbstractPDMat) = PDMat(a.mat + full(b))
+(a::PDiagMat, b::AbstractPDMat) = PDMat(_adddiag!(full(b), a.diag))
+(a::ScalMat, b::AbstractPDMat) = PDMat(_adddiag!(full(b), a.value))
+(a::PDSparseMat, b::AbstractPDMat) = PDMat(a.mat + full(b))

+(a::PDMat, b::PDMat) = PDMat(a.mat + b.mat)
+(a::PDMat, b::PDiagMat) = PDMat(_adddiag(a.mat, b.diag))
+(a::PDMat, b::ScalMat) = PDMat(_adddiag(a.mat, b.value))
+(a::PDMat, b::PDSparseMat) = PDMat(a.mat + b.mat)

+(a::PDiagMat, b::PDMat) = PDMat(_adddiag(b.mat, a.diag))
+(a::PDiagMat, b::PDiagMat) = PDiagMat(a.diag + b.diag)
+(a::PDiagMat, b::ScalMat) = PDiagMat(a.diag .+ b.value)
+(a::PDiagMat, b::PDSparseMat) = PDSparseMat(_adddiag(b.mat, a.diag))

+(a::ScalMat, b::PDMat) = PDMat(_adddiag(b.mat, a.value))
+(a::ScalMat, b::PDiagMat) = PDiagMat(a.value .+ b.diag)
+(a::ScalMat, b::ScalMat) = ScalMat(a.dim, a.value + b.value)
+(a::ScalMat, b::PDSparseMat) = PDSparseMat(_adddiag(b.mat, a.value))

+(a::PDSparseMat, b::PDMat) = PDMat(full(a) + b.mat)
+(a::PDSparseMat, b::PDiagMat) = PDSparseMat(_adddiag(a.mat, b.diag))
+(a::PDSparseMat, b::ScalMat) = PDSparseMat(_adddiag(a.mat, b.value))
+(a::PDSparseMat, b::PDSparseMat) = PDSparseMat(a.mat + b.mat)


# between pdmat and uniformscaling (multiple of identity)

+(a::AbstractPDMat, b::UniformScaling) = a + ScalMat(dim(a), convert(Float64, b.λ))
+(a::UniformScaling, b::AbstractPDMat) = ScalMat(dim(b), convert(Float64, a.λ)) + b

pdadd(a::PDMat, b::AbstractPDMat, c::Float64) = PDMat(a.mat + full(b * c))
pdadd(a::PDiagMat, b::AbstractPDMat, c::Float64) = PDMat(_adddiag!(full(b * c), a.diag, 1.0))
pdadd(a::ScalMat, b::AbstractPDMat, c::Float64) = PDMat(_adddiag!(full(b * c), a.value))
pdadd(a::PDSparseMat, b::AbstractPDMat, c::Float64) = PDMat(a.mat + full(b * c))

pdadd(a::PDMat, b::PDMat, c::Float64) = PDMat(a.mat + b.mat * c)
pdadd(a::PDMat, b::PDiagMat, c::Float64) = PDMat(_adddiag(a.mat, b.diag, c))
pdadd(a::PDMat, b::ScalMat, c::Float64) = PDMat(_adddiag(a.mat, b.value * c))
pdadd(a::PDMat, b::PDSparseMat, c::Float64) = PDMat(a.mat + b.mat * c)

pdadd(a::PDiagMat, b::PDMat, c::Float64) = PDMat(_adddiag!(b.mat * c, a.diag, 1.0))
pdadd(a::PDiagMat, b::PDiagMat, c::Float64) = PDiagMat(a.diag + b.diag * c)
pdadd(a::PDiagMat, b::ScalMat, c::Float64) = PDiagMat(a.diag .+ b.value * c)
pdadd(a::PDiagMat, b::PDSparseMat, c::Float64) = PDSparseMat(_adddiag!(b.mat * c, a.diag, 1.0))

pdadd(a::ScalMat, b::PDMat, c::Float64) = PDMat(_adddiag!(b.mat * c, a.value))
pdadd(a::ScalMat, b::PDiagMat, c::Float64) = PDiagMat(a.value .+ b.diag * c)
pdadd(a::ScalMat, b::ScalMat, c::Float64) = ScalMat(a.dim, a.value + b.value * c)
pdadd(a::ScalMat, b::PDSparseMat, c::Float64) = PDSparseMat(_adddiag!(b.mat * c, a.value))

pdadd(a::PDSparseMat, b::PDMat, c::Float64) = PDMat(a.mat + b.mat * c)
pdadd(a::PDSparseMat, b::PDiagMat, c::Float64) = PDSparseMat(_adddiag(a.mat, b.diag, c))
pdadd(a::PDSparseMat, b::ScalMat, c::Float64) = PDSparseMat(_adddiag(a.mat, b.value * c))
pdadd(a::PDSparseMat, b::PDSparseMat, c::Float64) = PDSparseMat(a.mat + b.mat * c)
