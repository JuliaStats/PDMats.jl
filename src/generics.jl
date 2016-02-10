# Generic functions (on top of the type-specific implementations)

## Basic functions

Base.eltype{T<:AbstractFloat}(a::AbstractPDMat{T}) = T
Base.ndims(a::AbstractPDMat) = 2
Base.size(a::AbstractPDMat) = (dim(a), dim(a))
Base.size(a::AbstractPDMat, i::Integer) = 1 <= i <= 2 ? dim(a) : 1
Base.length(a::AbstractPDMat) = abs2(dim(a))

## arithmetics

pdadd!{T<:AbstractFloat}(r::Matrix{T}, a::Matrix{T}, b::AbstractPDMat{T}) = pdadd!(r, a, b, one(T))

pdadd!{T<:AbstractFloat}(a::Matrix{T}, b::AbstractPDMat{T}, c::T) = pdadd!(a, a, b, c)
pdadd!{T<:AbstractFloat}(a::Matrix{T}, b::AbstractPDMat{T}) = pdadd!(a, a, b, one(T))

pdadd{T<:AbstractFloat}(a::Matrix{T}, b::AbstractPDMat{T}, c::T) = pdadd!(similar(a), a, b, c)
pdadd{T<:AbstractFloat}(a::Matrix{T}, b::AbstractPDMat{T}) = pdadd!(similar(a), a, b, one(T))

+{T<:AbstractFloat}(a::Matrix{T}, b::AbstractPDMat{T}) = pdadd(a, b)
+{T<:AbstractFloat}(a::AbstractPDMat{T}, b::Matrix{T}) = pdadd(b, a)

*{T<:AbstractFloat}(a::AbstractPDMat{T}, c::T) = a * c
*{T<:AbstractFloat}(c::T, a::AbstractPDMat{T}) = a * c
/{T<:AbstractFloat}(a::AbstractPDMat{T}, c::T) = a * inv(c)


## whiten and unwhiten

whiten!{T<:AbstractFloat}(a::AbstractPDMat{T}, x::StridedVecOrMat{T}) = whiten!(x, a, x)
unwhiten!{T<:AbstractFloat}(a::AbstractPDMat{T}, x::StridedVecOrMat{T}) = unwhiten!(x, a, x)

whiten{T<:AbstractFloat}(a::AbstractPDMat{T}, x::StridedVecOrMat{T}) = whiten!(similar(x), a, x)
unwhiten{T<:AbstractFloat}(a::AbstractPDMat{T}, x::StridedVecOrMat{T}) = unwhiten!(similar(x), a, x)


## quad

function quad{T<:AbstractFloat}(a::AbstractPDMat{T}, x::StridedMatrix{T})
    @check_argdims dim(a) == size(x, 1)
    quad!(Array(T, size(x,2)), a, x)
end

function invquad{T<:AbstractFloat}(a::AbstractPDMat{T}, x::StridedMatrix{T})
    @check_argdims dim(a) == size(x, 1)
    invquad!(Array(T, size(x,2)), a, x)
end
