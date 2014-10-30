# Generic functions (on top of the type-specific implementations)

## Basic functions

Base.eltype(a::AbstractPDMat) = Float64
Base.ndims(a::AbstractPDMat) = 2
Base.size(a::AbstractPDMat) = (dim(a), dim(a))
Base.size(a::AbstractPDMat, i::Integer) = 1 <= i <= 2 ? dim(a) : 1
Base.length(a::AbstractPDMat) = abs2(dim(a))


* (c::Float64, a::AbstractPDMat) = a * c
/ (a::AbstractPDMat, c::Float64) = a * inv(c)

function quad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    r = Array(Float64, size(x,2))
    quad!(r, a, x)
    r
end

function invquad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    r = Array(Float64, size(x,2))
    invquad!(r, a, x)
    r
end

