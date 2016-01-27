# Useful utilities to support internal implementation


macro check_argdims(cond)
    quote
        ($(cond)) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end

_rcopy!(r::DenseVecOrMat, x::DenseVecOrMat) = (is(r, x) || copy!(r, x); r)


@compat function _addscal!{T<:AbstractFloat}(r::Matrix{T}, a::Matrix{T}, b::Union{Matrix{T}, SparseMatrixCSC{T}}, c::T)
    if c == one(T)
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

@compat function _adddiag!{T<:AbstractFloat}(a::Union{Matrix{T}, SparseMatrixCSC{T}}, v::T)
    n = size(a, 1)
    for i = 1:n
        @inbounds a[i,i] += v
    end
    return a
end

@compat function _adddiag!{T<:AbstractFloat}(a::Union{Matrix{T}, SparseMatrixCSC{T}}, v::Vector{T}, c::T)
    n = size(a, 1)
    @check_argdims length(v) == n
    if c == one(T)
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

@compat _adddiag{T<:AbstractFloat}(a::Union{Matrix{T}, SparseMatrixCSC{T}}, v::T) = _adddiag!(copy(a), v)
@compat _adddiag{T<:AbstractFloat}(a::Union{Matrix{T}, SparseMatrixCSC{T}}, v::Vector{T}, c::T) = _adddiag!(copy(a), v, c)
@compat _adddiag{T<:AbstractFloat}(a::Union{Matrix{T}, SparseMatrixCSC{T}}, v::Vector{T}) = _adddiag!(copy(a), v, one(T))

function wsumsq{T<:AbstractFloat}(w::AbstractVector{T}, a::AbstractVector{T})
    @check_argdims(length(a) == length(w))
    s = zero(T)
    for i = 1:length(a)
        @inbounds s += abs2(a[i]) * w[i]
    end
    return s
end

function colwise_dot!{T<:AbstractFloat}(r::AbstractArray{T}, a::DenseMatrix{T}, b::DenseMatrix{T})
    n = length(r)
    @check_argdims n == size(a, 2) == size(b, 2) && size(a, 1) == size(b, 1)
    for i = 1:n
        r[i] = dot(view(a,:,i), view(b,:,i))
    end
    return r
end

function colwise_sumsq!{T<:AbstractFloat}(r::AbstractArray{T}, a::DenseMatrix{T}, c::T)
    n = length(r)
    @check_argdims n == size(a, 2)
    for i = 1:n
        r[i] = sumabs2(view(a,:,i)) * c
    end
    return r
end
