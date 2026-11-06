# Generic functions (on top of the type-specific implementations)

## constructors
AbstractPDMat(A::AbstractPDMat) = A
AbstractPDMat(A::AbstractMatrix) = PDMat(A)

## convert
Base.convert(::Type{AbstractMatrix{T}}, a::AbstractPDMat) where {T <: Real} = convert(AbstractPDMat{T}, a)
Base.convert(::Type{AbstractArray{T}}, a::AbstractPDMat) where {T <: Real} = convert(AbstractMatrix{T}, a)

## arithmetics

pdadd!(r::Matrix, a::Matrix, b::AbstractPDMat{T}) where {T <: Real} = pdadd!(r, a, b, one(T))

pdadd!(a::Matrix, b::AbstractPDMat, c) = pdadd!(a, a, b, c)
pdadd!(a::Matrix, b::AbstractPDMat{T}) where {T <: Real} = pdadd!(a, a, b, one(T))

pdadd(a::Matrix{T}, b::AbstractPDMat{S}, c::R) where {T <: Real, S <: Real, R <: Real} = pdadd!(similar(a, promote_type(T, S, R)), a, b, c)
pdadd(a::Matrix{T}, b::AbstractPDMat{S}) where {T <: Real, S <: Real} = pdadd!(similar(a, promote_type(T, S)), a, b, one(T))

+(a::Matrix, b::AbstractPDMat) = pdadd(a, b)
+(a::AbstractPDMat, b::Matrix) = pdadd(b, a)

*(c::T, a::AbstractPDMat) where {T <: Real} = a * c
/(a::AbstractPDMat, c::T) where {T <: Real} = a * inv(c)
Base.kron(A::AbstractPDMat, B::AbstractPDMat) = PDMat(kron(Matrix(A), Matrix(B)))

# Avoid the intermediate only for same-precision finite floating-point vectors.
# Multiply the weight first, as in the original matrix-vector product: squaring
# first can overflow or underflow. Other inputs retain the original promotion.
function *(
        x::Union{Adjoint{<:Any, <:AbstractVector{<:Real}}, Transpose{<:Any, <:AbstractVector{<:Real}}},
        a::Union{PDiagMat, ScalMat}, y::AbstractVector{<:Real}
    )
    T = eltype(y)
    if parent(x) === y && T === eltype(a) && (T === Float32 || T === Float64) &&
            axes(y, 1) == Base.OneTo(length(y)) &&
            (a isa ScalMat || axes(a.diag, 1) == Base.OneTo(a.dim)) && all(isfinite, y)
        result = _ordered_quadratic(a, y)
        result === nothing || return result
    end
    return (x * a) * y
end

function _ordered_quadratic(a::Union{PDiagMat{T}, ScalMat{T}}, x::AbstractVector{T}) where {T <: Union{Float32, Float64}}
    Base.require_one_based_indexing(x)
    a isa PDiagMat && Base.require_one_based_indexing(a.diag)
    @check_argdims a.dim == length(x)
    # Keep the first product in the original operand precision. Widen Float32's
    # final product and compensate the sum so small contributions are retained.
    total = 0.0
    correction = 0.0
    for i in eachindex(x)
        weight = a isa PDiagMat ? a.diag[i] : a.value
        scaled = weight * x[i]
        term = Float64(scaled) * Float64(x[i])
        # Preserve the original BLAS path at exceptional/underflow boundaries.
        if !isfinite(term) || issubnormal(term) ||
                (iszero(term) && !iszero(scaled) && !iszero(x[i]))
            return nothing
        end
        next = total + term
        isfinite(next) || return nothing
        correction += abs(total) >= abs(term) ? (total - next) + term : (term - next) + total
        isfinite(correction) || return nothing
        total = next
    end
    combined = total + correction
    result = T(combined)
    if !isfinite(result) || issubnormal(result) || (iszero(result) && !iszero(combined))
        return nothing
    end
    return result
end

# LinearAlgebra
LinearAlgebra.isposdef(::AbstractPDMat) = true
LinearAlgebra.ishermitian(::AbstractPDMat) = true
LinearAlgebra.checksquare(a::AbstractPDMat) = size(a, 1)
LinearAlgebra.ldiv!(A::AbstractPDMat, B::AbstractVecOrMat) = ldiv!(cholesky(A), B)

## whiten and unwhiten

"""
    whiten(a::AbstractMatrix, x::AbstractVecOrMat)
    whiten!(a::AbstractMatrix, x::AbstractVecOrMat)
    whiten!(r::AbstractVecOrMat, a::AbstractPDMat, x::AbstractVecOrMat)

Allocating and in-place versions of the `whiten`ing transform defined by `a` applied to `x`.

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
PDMat{Float64, Matrix{Float64}, Cholesky{Float64, Matrix{Float64}}}([4.0 10.0; 10.0 30.0], Cholesky{Float64, Matrix{Float64}}([2.0 5.0; 10.0 2.23607], 'U', 0))

julia> W = whiten(a, X)
2×4 Array{Float64,2}:
  0.5       0.5       0.5       0.5
 -0.67082  -0.223607  0.223607  0.67082

julia> W * W'
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  1.0
```

`whiten` is the inverse transformation of [`unwhiten`](@ref).

See also [`invwhiten`](@ref) and [`invunwhiten`](ref)
"""
whiten(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = whiten(AbstractPDMat(a), x)

"""
    invwhiten(a::AbstractMatrix, x::AbstractVecOrMat)
    invwhiten!(a::AbstractMatrix, x::AbstractVecOrMat)
    invwhiten!(r::AbstractVecOrMat, a::AbstractPDMat, x::AbstractVecOrMat)

Allocating and in-place versions of the `whiten`ing transform defined by `inv(a)` applied to `x`.

If the precision of `x` is `a` then the covariance of the result will be `I`.
The name `invwhiten` indicates that this function transforms a correlated multivariate random
variable to "white noise".

`invwhiten` is the inverse transformation of [`invunwhiten`](@ref).

See also [`whiten`](@ref) and [`unwhiten`](@ref)
"""
invwhiten(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = invwhiten(AbstractPDMat(a), x)

"""
    unwhiten(a::AbstractMatrix, x::AbstractVecOrMat)
    unwhiten!(a::AbstractMatrix, x::AbstractVecOrMat)
    unwhiten!(r::AbstractVecOrMat, a::AbstractPDMat, x::AbstractVecOrMat)

Allocating and in-place versions of the `unwhiten`ing transform defined by `a` applied to `x`.

If the covariance of `x` is `I` then the covariance of the result will be `a`.
The name `unwhiten` indicates that this function transforms an uncorrelated "white noise" signal
into a signal with the given covariance.

`unwhiten` is the inverse transformation of [`whiten`](@ref).

See also [`invwhiten`](@ref) and [`invunwhiten`](@ref)
"""
unwhiten(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = unwhiten(AbstractPDMat(a), x)

"""
    invunwhiten(a::AbstractMatrix, x::AbstractVecOrMat)
    invunwhiten!(a::AbstractMatrix, x::AbstractVecOrMat)
    invunwhiten!(r::AbstractVecOrMat, a::AbstractPDMat, x::AbstractVecOrMat)

Allocating and in-place versions of the `unwhiten`ing transform defined by `inv(a)` applied to `x`.

If the covariance of `x` is `I` then the precision of the result will be `a`.
The name `invunwhiten` indicates that this function transforms an uncorrelated "white noise" signal
into a signal with the given covariance.

`invunwhiten` is the inverse transformation of [`invwhiten`](@ref).

See also [`whiten`](@ref) and [`unwhiten`](@ref)
"""
invunwhiten(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = invunwhiten(AbstractPDMat(a), x)

whiten!(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = whiten!(x, a, x)
invwhiten!(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = invwhiten!(x, a, x)
unwhiten!(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = unwhiten!(x, a, x)
invunwhiten!(a::AbstractMatrix{<:Real}, x::AbstractVecOrMat) = invunwhiten!(x, a, x)

function whiten!(r::AbstractVecOrMat, a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return whiten!(r, AbstractPDMat(a), x)
end
function invwhiten!(r::AbstractVecOrMat, a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return invwhiten!(r, AbstractPDMat(a), x)
end
function unwhiten!(r::AbstractVecOrMat, a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return unwhiten!(r, AbstractPDMat(a), x)
end
function invunwhiten!(r::AbstractVecOrMat, a::AbstractMatrix{<:Real}, x::AbstractVecOrMat)
    return invunwhiten!(r, AbstractPDMat(a), x)
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
