# deprecated functions

using Base: @deprecate

@deprecate add!(a::Matrix, b::AbstractPDMat) pdadd!(a, b)

@deprecate add_scal!(a::Matrix, b::AbstractPDMat, c::Real) pdadd!(a, b, c)

@deprecate add_scal(a::Matrix, b::AbstractPDMat, c::Real) pdadd(a, b, c)

@deprecate full(x::AbstractPDMat) Matrix(x)

@deprecate CholType Cholesky

@deprecate PDiagMat{T,S}(d::Int,v::AbstractVector,inv_v::AbstractVector) where {T,S} PDiagMat{T,S}(d::Int,v::AbstractVector) where {T,S}
