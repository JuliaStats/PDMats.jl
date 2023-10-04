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

@deprecate PDMat{T,S}(d::Int, m::AbstractMatrix{T}, c::Cholesky{T,S}) where {T,S} PDMat{T,S}(m, c)

@deprecate PDiagMat(dim::Int, diag::AbstractVector{<:Real}) PDiagMat(diag)
@deprecate PDiagMat{T,V}(dim, diag) where {T<:Real, V<:AbstractVector{T}} PDiagMat{T,V}(diag)

