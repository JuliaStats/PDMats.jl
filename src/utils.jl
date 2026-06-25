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

# Non-mutating diagonal addition: `a + diagonal` (optionally scaled by `c`). These
# work for dense, sparse, and immutable (e.g. static) matrices, and for dense
# matrices cost a single allocation just like the in-place path above (which is kept
# only for the truly in-place `pdadd!`).
_adddiag(a::AbstractMatrix, v::Real) = a + v * I

function _adddiag(a::AbstractMatrix, v::AbstractVector)
    @check_argdims eachindex(v) == axes(a, 1) == axes(a, 2)
    return a + Diagonal(v)
end

function _adddiag(a::AbstractMatrix, v::AbstractVector, c::Real)
    @check_argdims eachindex(v) == axes(a, 1) == axes(a, 2)
    return a .+ c .* Diagonal(v)
end

# As above, but the coefficient scales the matrix: `c * a + diagonal`. The scalar
# variant keeps an in-place fast path for mutable storage, since there is no clean
# single-allocation broadcast for adding a scalar to the diagonal only; other matrix
# types fall back to `muladd`, allowing fused implementations where available.
function _scaleadddiag(a::AbstractMatrix, c::Real, v::AbstractVector)
    @check_argdims eachindex(v) == axes(a, 1) == axes(a, 2)
    return c .* a .+ Diagonal(v)
end
_scaleadddiag(a::AbstractMatrix, c::Real, v::Real) = muladd(c, a, v * I)
_scaleadddiag(a::Union{Matrix, SparseMatrixCSC}, c::Real, v::Real) = _adddiag!(a * c, v)


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
