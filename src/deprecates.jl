# deprecated functions

using Base: @deprecate

@deprecate add!(a::Matrix, b::AbstractPDMat) pdadd!(a, b)

@deprecate add_scal!(a::Matrix, b::AbstractPDMat, c::Real) pdadd!(a, b, c)

@deprecate add_scal(a::Matrix, b::AbstractPDMat, c::Real) pdadd(a, b, c)

@deprecate full(x::AbstractPDMat) Matrix(x)
