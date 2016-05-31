# Generic functions (on top of the type-specific implementations)

## Basic functions

Base.eltype{T<:Real}(a::AbstractPDMat{T}) = T
Base.ndims(a::AbstractPDMat) = 2
Base.size(a::AbstractPDMat) = (dim(a), dim(a))
Base.size(a::AbstractPDMat, i::Integer) = 1 <= i <= 2 ? dim(a) : 1
Base.length(a::AbstractPDMat) = abs2(dim(a))

## arithmetics

pdadd!{T<:Real}(r::Matrix{T}, a::Matrix{T}, b::AbstractPDMat{T}) = pdadd!(r, a, b, one(T))

pdadd!(a::Matrix, b::AbstractPDMat, c) = pdadd!(a, a, b, c)
pdadd!{T<:Real}(a::Matrix{T}, b::AbstractPDMat{T}) = pdadd!(a, a, b, one(T))

pdadd(a::Matrix, b::AbstractPDMat, c) = pdadd!(similar(a), a, b, c)
pdadd{T<:Real}(a::Matrix{T}, b::AbstractPDMat{T}) = pdadd!(similar(a), a, b, one(T))

+(a::Matrix, b::AbstractPDMat) = pdadd(a, b)
+(a::AbstractPDMat, b::Matrix) = pdadd(b, a)

*{T<:Real}(a::AbstractPDMat, c::T) = a * c
*{T<:Real}(c::T, a::AbstractPDMat) = a * c
/{T<:Real}(a::AbstractPDMat, c::T) = a * inv(c)


## whiten and unwhiten

whiten!(a::AbstractPDMat, x::StridedVecOrMat) = whiten!(x, a, x)
unwhiten!(a::AbstractPDMat, x::StridedVecOrMat) = unwhiten!(x, a, x)

whiten(a::AbstractPDMat, x::StridedVecOrMat) = whiten!(similar(x), a, x)
unwhiten(a::AbstractPDMat, x::StridedVecOrMat) = unwhiten!(similar(x), a, x)


## quad

function quad{T<:Real}(a::AbstractPDMat{T}, x::StridedMatrix{T})
    @check_argdims dim(a) == size(x, 1)
    quad!(Array(T, size(x,2)), a, x)
end

function invquad{T<:Real}(a::AbstractPDMat{T}, x::StridedMatrix{T})
    @check_argdims dim(a) == size(x, 1)
    invquad!(Array(T, size(x,2)), a, x)
end
