# Useful utilities to support internal implementation


macro check_argdims(cond)
    return quote
        ($(esc(cond))) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end

function _addscal!(r::Matrix, a::Matrix, b::Union{Matrix, SparseMatrixCSC}, c::Real)
    if c == one(c)
        for i in eachindex(a)
            @inbounds r[i] = a[i] + b[i]
        end
    else
        for i in eachindex(a)
            @inbounds r[i] = a[i] + b[i] * c
        end
    end
    return r
end

function _adddiag!(a::Union{Matrix, SparseMatrixCSC}, v::Real)
    for i in diagind(a)
        @inbounds a[i] += v
    end
    return a
end

function _adddiag!(a::Union{Matrix, SparseMatrixCSC}, v::AbstractVector, c::Real)
    @check_argdims eachindex(v) == axes(a, 1) == axes(a, 2)
    if c == one(c)
        for i in eachindex(v)
            @inbounds a[i, i] += v[i]
        end
    else
        for i in eachindex(v)
            @inbounds a[i, i] += v[i] * c
        end
    end
    return a
end

_adddiag(a::Union{Matrix, SparseMatrixCSC}, v::Real) = _adddiag!(copy(a), v)
_adddiag(a::Union{Matrix, SparseMatrixCSC}, v::AbstractVector, c::Real) = _adddiag!(copy(a), v, c)
_adddiag(a::Union{Matrix, SparseMatrixCSC}, v::AbstractVector{T}) where {T <: Real} = _adddiag!(copy(a), v, one(T))


function wsumsq(w::AbstractVector, a::AbstractVector)
    @check_argdims(eachindex(a) == eachindex(w))
    s = zero(promote_type(eltype(w), eltype(a)))
    for i in eachindex(w)
        @inbounds s += abs2(a[i]) * w[i]
    end
    return s
end

function invwsumsq(w::AbstractVector, a::AbstractVector)
    @check_argdims(eachindex(a) == eachindex(w))
    s = zero(zero(eltype(a)) / zero(eltype(w)))
    for i in eachindex(w)
        @inbounds s += abs2(a[i]) / w[i]
    end
    return s
end

function colwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    @check_argdims(axes(a) == axes(b))
    @check_argdims(axes(a, 2) == eachindex(r))
    for j in axes(a, 2)
        v = zero(promote_type(eltype(a), eltype(b)))
        @simd for i in axes(a, 1)
            @inbounds v += a[i, j] * b[i, j]
        end
        r[j] = v
    end
    return r
end

function colwise_sumsq!(r::AbstractArray, a::AbstractMatrix, c::Real)
    @check_argdims(eachindex(r) == axes(a, 2))
    for j in axes(a, 2)
        v = zero(eltype(a))
        @simd for i in axes(a, 1)
            @inbounds v += abs2(a[i, j])
        end
        r[j] = v * c
    end
    return r
end

function colwise_sumsqinv!(r::AbstractArray, a::AbstractMatrix, c::Real)
    @check_argdims(eachindex(r) == axes(a, 2))
    for j in axes(a, 2)
        v = zero(eltype(a))
        @simd for i in axes(a, 1)
            @inbounds v += abs2(a[i, j])
        end
        r[j] = v / c
    end
    return r
end
