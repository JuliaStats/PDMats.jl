# Useful utilities to support internal implementation


macro check_argdims(cond)
    quote
        ($(cond)) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end

_rcopy!(r::DenseVecOrMat, x::DenseVecOrMat) = (is(r, x) || copy!(r, x); r)


function _addscal!(r::Matrix, a::Matrix, b::Union(Matrix, SparseMatrixCSC), c::Real)
    if c == 1.0
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

function _adddiag!(a::Matrix, v::Real)
    n = size(a, 1)
    for i = 1:n
        @inbounds a[i,i] += v
    end
    return a
end

function _adddiag!(a::Matrix, v::Vector, c::Real)
    n = size(a, 1)
    @check_argdims length(v) == n
    if c == 1.0
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

_adddiag(a::Matrix, v::Real) = _adddiag!(copy(a), v)
_adddiag(a::Matrix, v::Vector, c::Real) = _adddiag!(copy(a), v, c)
_adddiag(a::Matrix, v::Vector) = _adddiag!(copy(a), v, 1.0)

function wsumsq(w::AbstractVector, a::AbstractVector)
    @check_argdims(length(a) == length(w))
    s = 0.
    for i = 1:length(a)
        @inbounds s += abs2(a[i]) * w[i]
    end
    return s
end

function colwise_dot!(r::AbstractArray, a::DenseMatrix, b::DenseMatrix)
    n = length(r)
    @check_argdims n == size(a, 2) == size(b, 2) && size(a, 1) == size(b, 1)
    for i = 1:n
        r[i] = dot(view(a,:,i), view(b,:,i))
    end
    return r
end

function colwise_sumsq!(r::AbstractArray, a::DenseMatrix, c::Real)
    n = length(r)
    @check_argdims n == size(a, 2)
    for i = 1:n
        r[i] = sumabs2(view(a,:,i)) * c
    end
    return r
end
