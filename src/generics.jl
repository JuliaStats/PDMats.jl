# Generic functions (on top of the type-specific implementations)

## Basic functions

Base.eltype(a::AbstractPDMat) = Float64
Base.ndims(a::AbstractPDMat) = 2
Base.size(a::AbstractPDMat) = (dim(a), dim(a))
Base.size(a::AbstractPDMat, i::Integer) = 1 <= i <= 2 ? dim(a) : 1
Base.length(a::AbstractPDMat) = abs2(dim(a))

## arithmetics

pdadd!(r::Matrix{Float64}, a::Matrix{Float64}, b::AbstractPDMat) = pdadd!(r, a, b, 1.0)

pdadd!(a::Matrix{Float64}, b::AbstractPDMat, c::Real) = pdadd!(a, a, b, float64(c))
pdadd!(a::Matrix{Float64}, b::AbstractPDMat) = pdadd!(a, a, b, 1.0)

pdadd(a::Matrix{Float64}, b::AbstractPDMat, c::Real) = pdadd!(similar(a), a, b, float64(c))
pdadd(a::Matrix{Float64}, b::AbstractPDMat) = pdadd!(similar(a), a, b, 1.0)

+ (a::Matrix{Float64}, b::AbstractPDMat) = pdadd(a, b)
+ (a::AbstractPDMat, b::Matrix{Float64}) = pdadd(b, a)

* (a::AbstractPDMat, c::Real) = a * float64(c)
* (c::Real, a::AbstractPDMat) = a * float64(c)
/ (a::AbstractPDMat, c::Real) = a * float64(inv(c))


## whiten and unwhiten

whiten!(a::AbstractPDMat, x::DenseVecOrMat{Float64}) = whiten!(x, a, x)
unwhiten!(a::AbstractPDMat, x::DenseVecOrMat{Float64}) = unwhiten!(x, a, x)

whiten(a::AbstractPDMat, x::DenseVecOrMat{Float64}) = whiten!(similar(x), a, x)
unwhiten(a::AbstractPDMat, x::DenseVecOrMat{Float64}) = unwhiten!(similar(x), a, x)


## quad

function quad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    quad!(Array(Float64, size(x,2)), a, x)
end

function invquad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    invquad!(Array(Float64, size(x,2)), a, x)
end

