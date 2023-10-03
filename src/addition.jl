
# between pdmat and pdmat

+(a::PDMat, b::AbstractPDMat) = PDMat(a.mat + Matrix(b))
+(a::PDiagMat, b::AbstractPDMat) = PDMat(_adddiag!(Matrix(b), a.diag, true))
+(a::ScalMat, b::AbstractPDMat) = PDMat(_adddiag!(Matrix(b), a.value))
if HAVE_CHOLMOD
    +(a::PDSparseMat, b::AbstractPDMat) = PDMat(a.mat + Matrix(b))
end

+(a::PDMat, b::PDMat) = PDMat(a.mat + b.mat)
+(a::PDMat, b::PDiagMat) = PDMat(_adddiag(a.mat, b.diag))
+(a::PDMat, b::ScalMat) = PDMat(_adddiag(a.mat, b.value))
if HAVE_CHOLMOD
    +(a::PDMat, b::PDSparseMat) = PDMat(a.mat + b.mat)
end

+(a::PDiagMat, b::PDMat) = PDMat(_adddiag(b.mat, a.diag))
+(a::PDiagMat, b::PDiagMat) = PDiagMat(a.diag + b.diag)
+(a::PDiagMat, b::ScalMat) = PDiagMat(a.diag .+ b.value)
if HAVE_CHOLMOD
    +(a::PDiagMat, b::PDSparseMat) = PDSparseMat(_adddiag(b.mat, a.diag))
end

+(a::ScalMat, b::PDMat) = PDMat(_adddiag(b.mat, a.value))
+(a::ScalMat, b::PDiagMat) = PDiagMat(a.value .+ b.diag)
+(a::ScalMat, b::ScalMat) = ScalMat(a.dim, a.value + b.value)
if HAVE_CHOLMOD
    +(a::ScalMat, b::PDSparseMat) = PDSparseMat(_adddiag(b.mat, a.value))
end

if HAVE_CHOLMOD
    +(a::PDSparseMat, b::PDMat) = PDMat(Matrix(a) + b.mat)
    +(a::PDSparseMat, b::PDiagMat) = PDSparseMat(_adddiag(a.mat, b.diag))
    +(a::PDSparseMat, b::ScalMat) = PDSparseMat(_adddiag(a.mat, b.value))
    +(a::PDSparseMat, b::PDSparseMat) = PDSparseMat(a.mat + b.mat)
end


# between pdmat and uniformscaling (multiple of identity)

+(a::AbstractPDMat, b::UniformScaling) = a + ScalMat(a.dim, b.λ)
+(a::UniformScaling, b::AbstractPDMat) = ScalMat(b.dim, a.λ) + b

pdadd(a::PDMat, b::AbstractPDMat, c::Real) = PDMat(a.mat + Matrix(b * c))
pdadd(a::PDiagMat, b::AbstractPDMat, c::Real) = PDMat(_adddiag!(Matrix(b * c), a.diag, one(c)))
pdadd(a::ScalMat, b::AbstractPDMat, c::Real) = PDMat(_adddiag!(Matrix(b * c), a.value))
if HAVE_CHOLMOD
    pdadd(a::PDSparseMat, b::AbstractPDMat, c::Real) = PDMat(a.mat + Matrix(b * c))
end

pdadd(a::PDMat, b::PDMat, c::Real) = PDMat(a.mat + b.mat * c)
pdadd(a::PDMat, b::PDiagMat, c::Real) = PDMat(_adddiag(a.mat, b.diag, c))
pdadd(a::PDMat, b::ScalMat, c::Real) = PDMat(_adddiag(a.mat, b.value * c))
if HAVE_CHOLMOD
    pdadd(a::PDMat, b::PDSparseMat, c::Real) = PDMat(a.mat + b.mat * c)
end

pdadd(a::PDiagMat, b::PDMat, c::Real) = PDMat(_adddiag!(b.mat * c, a.diag, one(c)))
pdadd(a::PDiagMat, b::PDiagMat, c::Real) = PDiagMat(a.diag + b.diag * c)
pdadd(a::PDiagMat, b::ScalMat, c::Real) = PDiagMat(a.diag .+ b.value * c)
if HAVE_CHOLMOD
    pdadd(a::PDiagMat, b::PDSparseMat, c::Real) = PDSparseMat(_adddiag!(b.mat * c, a.diag, one(c)))
end

pdadd(a::ScalMat, b::PDMat, c::Real) = PDMat(_adddiag!(b.mat * c, a.value))
pdadd(a::ScalMat, b::PDiagMat, c::Real) = PDiagMat(a.value .+ b.diag * c)
pdadd(a::ScalMat, b::ScalMat, c::Real) = ScalMat(a.dim, a.value + b.value * c)
if HAVE_CHOLMOD
    pdadd(a::ScalMat, b::PDSparseMat, c::Real) = PDSparseMat(_adddiag!(b.mat * c, a.value))
end

if HAVE_CHOLMOD
    pdadd(a::PDSparseMat, b::PDMat, c::Real) = PDMat(a.mat + b.mat * c)
    pdadd(a::PDSparseMat, b::PDiagMat, c::Real) = PDSparseMat(_adddiag(a.mat, b.diag, c))
    pdadd(a::PDSparseMat, b::ScalMat, c::Real) = PDSparseMat(_adddiag(a.mat, b.value * c))
    pdadd(a::PDSparseMat, b::PDSparseMat, c::Real) = PDSparseMat(a.mat + b.mat * c)
end
