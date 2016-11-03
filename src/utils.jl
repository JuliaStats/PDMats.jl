# Useful utilities to support internal implementation


macro check_argdims(cond)
    quote
        ($(cond)) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end

_rcopy!(r::StridedVecOrMat, x::StridedVecOrMat) = (r === x || copy!(r, x); r)


@compat function _addscal!(r::Matrix, a::Matrix, b::Union{Matrix, SparseMatrixCSC}, c::Real)
    if c == one(c)
        for i = 1:length(a)
            @inbounds r[i] = a[i] + b[i]
        end
    else
        for i = 1:length(a)
            @inbounds r[i] = a[i] + b[i] * c
        end
    end
    return r
end

@compat function _adddiag!(a::Union{Matrix, SparseMatrixCSC}, v::Real)
    n = size(a, 1)
    for i = 1:n
        @inbounds a[i,i] += v
    end
    return a
end

@compat function _adddiag!(a::Union{Matrix, SparseMatrixCSC}, v::Vector, c::Real)
    n = size(a, 1)
    @check_argdims length(v) == n
    if c == one(c)
        for i = 1:n
            @inbounds a[i,i] += v[i]
        end
    else
        for i = 1:n
            @inbounds a[i,i] += v[i] * c
        end
    end
    return a
end

@compat _adddiag(a::Union{Matrix, SparseMatrixCSC}, v::Real) = _adddiag!(copy(a), v)
@compat _adddiag(a::Union{Matrix, SparseMatrixCSC}, v::Vector, c::Real) = _adddiag!(copy(a), v, c)
@compat _adddiag{T<:Real}(a::Union{Matrix, SparseMatrixCSC}, v::Vector{T}) = _adddiag!(copy(a), v, one(T))

function wsumsq(w::AbstractVector, a::AbstractVector)
    @check_argdims(length(a) == length(w))
    s = zero(promote_type(eltype(w), eltype(a)))
    for i = 1:length(a)
        @inbounds s += abs2(a[i]) * w[i]
    end
    return s
end

function colwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    n = length(r)
    @check_argdims n == size(a, 2) == size(b, 2) && size(a, 1) == size(b, 1)
    for j = 1:n
        v = zero(promote_type(eltype(a), eltype(b)))
        @simd for i = 1:size(a, 1)
            @inbounds v += a[i, j]*b[i, j]
        end
        r[j] = v
    end
    return r
end

function colwise_sumsq!(r::AbstractArray, a::AbstractMatrix, c::Real)
    n = length(r)
    @check_argdims n == size(a, 2)
    for j = 1:n
        v = zero(eltype(a))
        @simd for i = 1:size(a, 1)
            @inbounds v += abs2(a[i, j])
        end
        r[j] = v*c
    end
    return r
end
