# Generic functions (on top of the type-specific implementations)

## constructors
AbstractPDMat(A::AbstractPDMat) = A
AbstractPDMat(A::AbstractMatrix) = PDMat(A)

## convert
Base.convert(::Type{AbstractMatrix{T}}, a::AbstractPDMat) where {T<:Real} = convert(AbstractPDMat{T}, a) 
Base.convert(::Type{AbstractArray{T}}, a::AbstractPDMat) where {T<:Real} = convert(AbstractMatrix{T}, a)

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
Base.kron(A::AbstractPDMat, B::AbstractPDMat) = PDMat(kron(Matrix(A), Matrix(B)))

# LinearAlgebra
LinearAlgebra.isposdef(::AbstractPDMat) = true
LinearAlgebra.ishermitian(::AbstractPDMat) = true
LinearAlgebra.checksquare(a::AbstractPDMat) = size(a, 1)

## whiten and unwhiten

"""
    whiten(a::AbstractMatrix, x::AbstractVecOrMat)
    unwhiten(a::AbstractMatrix, x::AbstractVecOrMat)
    unwhiten!(a::AbstractMatrix, x::AbstractVecOrMat)
    unwhiten!(r::AbstractVecOrMat, a::AbstractPDMat, x::AbstractVecOrMat)

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
whiten(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = whiten(AbstractPDMat(a), x)
unwhiten(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = unwhiten(AbstractPDMat(a), x)

whiten!(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = whiten!(x, a, x)
unwhiten!(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = unwhiten!(x, a, x)

function whiten!(r::AbstractVecOrMat, a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return whiten!(r, AbstractPDMat(a), x)
end
function unwhiten!(r::AbstractVecOrMat, a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return unwhiten!(r, AbstractPDMat(a), x)
end

## quad

"""
    quad(a::AbstractMatrix, x::AbstractVecOrMat)

Return the value of the quadratic form defined by `a` applied to `x`.

If `x` is a vector the quadratic form is `x' * a * x`.  If `x` is a matrix
the quadratic form is applied column-wise.
"""
function quad(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return quad(AbstractPDMat(a), x)
end

"""
    quad!(r::AbstractArray, a::AbstractMatrix, x::AbstractMatrix)

Overwrite `r` with the value of the quadratic form defined by `a` applied columnwise to `x`.
"""
function quad!(r::AbstractArray, a::AbstractMatrix{<:Real}, x::AbstractMatrix)
    return quad!(r, AbstractPDMat(a), x)
end

"""
    invquad(a::AbstractMatrix, x::AbstractVecOrMat)

Return the value of the quadratic form defined by `inv(a)` applied to `x`.

For most `PDMat` types this is done in a way that does not require evaluation of `inv(a)`.

If `x` is a vector the quadratic form is `x' * a * x`.  If `x` is a matrix
the quadratic form is applied column-wise.
"""
function invquad(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return invquad(AbstractPDMat(a), x)
end

"""
    invquad!(r::AbstractArray, a::AbstractMatrix, x::AbstractMatrix)

Overwrite `r` with the value of the quadratic form defined by `inv(a)` applied columnwise to `x`
"""
function invquad!(r::AbstractArray, a::AbstractMatrix{<:Real}, x::AbstractMatrix)
    return invquad!(r, AbstractPDMat(a), x)
end

