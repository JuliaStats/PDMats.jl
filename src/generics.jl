# Generic functions (on top of the type-specific implementations)

## Basic functions

Base.eltype(a::AbstractPDMat{T}) where {T<:Real} = T
Base.eltype(::Type{P}) where P<:AbstractPDMat{T} where T<:Real = T
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

whiten(a::AbstractPDMat, x::StridedVecOrMat) = whiten!(similar(x), a, x)
unwhiten(a::AbstractPDMat, x::StridedVecOrMat) = unwhiten!(similar(x), a, x)


## quad

function quad(a::AbstractPDMat{T}, x::StridedMatrix{S}) where {T<:Real, S<:Real}
    @check_argdims dim(a) == size(x, 1)
    quad!(Array{promote_type(T, S)}(uninitialized, size(x, 2)), a, x)
end

function invquad(a::AbstractPDMat{T}, x::StridedMatrix{S}) where {T<:Real, S<:Real}
    @check_argdims dim(a) == size(x, 1)
    invquad!(Array{promote_type(T, S)}(uninitialized, size(x, 2)), a, x)
end
