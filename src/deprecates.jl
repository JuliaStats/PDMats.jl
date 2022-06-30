# deprecated functions

using Base: @deprecate

@deprecate add!(a::Matrix, b::AbstractPDMat) pdadd!(a, b)

@deprecate add_scal!(a::Matrix, b::AbstractPDMat, c::Real) pdadd!(a, b, c)

@deprecate add_scal(a::Matrix, b::AbstractPDMat, c::Real) pdadd(a, b, c)

@deprecate full(x::AbstractPDMat) Matrix(x)

@deprecate CholType Cholesky

@deprecate ScalMat(d::Int, x::Real, inv_x::Real) ScalMat(d, x)
@deprecate PDiagMat(v::AbstractVector, inv_v::AbstractVector) PDiagMat(v)

@deprecate dim(a::AbstractMatrix) LinearAlgebra.checksquare(a)
