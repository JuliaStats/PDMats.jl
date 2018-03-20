# Generic functions (on top of the type-specific implementations)

## Basic functions

Base.eltype(a::AbstractPDMat{T}) where {T<:Real} = T
Base.eltype(::Type{AbstractPDMat{T}}) where {T<:Real} = T
Base.ndims(a::AbstractPDMat) = 2
Base.size(a::AbstractPDMat) = (dim(a), dim(a))
Base.size(a::AbstractPDMat, i::Integer) = 1 <= i <= 2 ? dim(a) : 1
Base.length(a::AbstractPDMat) = abs2(dim(a))

## arithmetics

pdadd!(r::Matrix, a::Matrix, b::AbstractPDMat{T}) where {T<:Real} = pdadd!(r, a, b, one(T))

pdadd!(a::Matrix, b::AbstractPDMat, c) = pdadd!(a, a, b, c)
pdadd!(a::Matrix, b::AbstractPDMat{T}) where {T<:Real} = pdadd!(a, a, b, one(T))

pdadd(a::Matrix{T}, b::AbstractPDMat{S}, c::R) where {T<:Real, S<:Real, R<:Real} = pdadd!(similar(a, promote_type(T, S, R)), a, b, c)
pdadd(a::Matrix{T}, b::AbstractPDMat{S}) where {T<:Real, S<:Real} = pdadd!(similar(a, promote_type(T, S)), a, b, one(T))

+(a::Matrix, b::AbstractPDMat) = pdadd(a, b)
+(a::AbstractPDMat, b::Matrix) = pdadd(b, a)

*(a::AbstractPDMat, c::T) where {T<:Real} = a * c
*(c::T, a::AbstractPDMat) where {T<:Real} = a * c
/(a::AbstractPDMat, c::T) where {T<:Real} = a * inv(c)


## whiten and unwhiten
whiten!(a::AbstractPDMat, x::StridedVecOrMat) = whiten!(x, a, x)
unwhiten!(a::AbstractPDMat, x::StridedVecOrMat) = unwhiten!(x, a, x)

"""
    whiten(a::AbstractPDMat, x::StridedVecOrMat)
    whiten!(a::AbstractPDMat, x::StridedVecOrMat)
    whiten!(r::StridedVecOrMat, a::AbstractPDMat, x::StridedVecOrMat)
    unwhiten(a::AbstractPDMat, x::StridedVecOrMat)
    unwhiten!(a::AbstractPDMat, x::StridedVecOrMat)
    unwhiten!(r::StridedVecOrMat, a::AbstractPDMat, x::StridedVecOrMat)

Allocating and in-place versions of the `whiten`ing transform (or its inverse) defined by `a` applied to `x`

If the covariance of `x` is `a` then the covariance of the result will be `I`.
The name `whiten` indicates that this function transforms a correlated multivariate random
variable to "white noise".

```jldoctest
julia> using PDMats

julia> X = vcat(ones(4)', (1:4)')
2×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  2.0  3.0  4.0

julia> a = PDMat(X * X')
PDMat{Float64,Array{Float64,2}}(2, [4.0 10.0; 10.0 30.0], Cholesky{Float64,Array{Float64,2}}([2.0 5.0; 10.0 2.23607], 'U', 0))

julia> W = whiten(a, X)
2×4 Array{Float64,2}:
  0.5       0.5       0.5       0.5    
 -0.67082  -0.223607  0.223607  0.67082

julia> W * W'
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  1.0
```
"""
whiten(a::AbstractPDMat, x::StridedVecOrMat) = whiten!(similar(x), a, x)
unwhiten(a::AbstractPDMat, x::StridedVecOrMat) = unwhiten!(similar(x), a, x)


## quad

"""
    quad(a::AbstractPDMat, x::StridedVecOrMat)

Return the value of the quadratic form defined by `a` applied to `x`

If `x` is a vector the quadratic form is `x' * a * x`.  If `x` is a matrix
the quadratic form is applied column-wise.
"""
function quad(a::AbstractPDMat{T}, x::StridedMatrix{S}) where {T<:Real, S<:Real}
    @check_argdims dim(a) == size(x, 1)
    quad!(Array{promote_type(T, S)}(undef, size(x,2)), a, x)
end


"""
    invquad(a::AbstractPDMat, x::StridedVecOrMat)

Return the value of the quadratic form defined by `inv(a)` applied to `x`.

For most `PDMat` types this is done in a way that does not require evaluation of `inv(a)`.

If `x` is a vector the quadratic form is `x' * a * x`.  If `x` is a matrix
the quadratic form is applied column-wise.
"""
function invquad(a::AbstractPDMat{T}, x::StridedMatrix{S}) where {T<:Real, S<:Real}
    @check_argdims dim(a) == size(x, 1)
    invquad!(Array{promote_type(T, S)}(undef, size(x,2)), a, x)
end
