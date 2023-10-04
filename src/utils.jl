# Useful utilities to support internal implementation


macro check_argdims(cond)
    quote
        ($(esc(cond))) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end

_rcopy!(r, x) = (r === x || copyto!(r, x); r)


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
            @inbounds a[i,i] += v[i]
        end
    else
        for i in eachindex(v)
            @inbounds a[i,i] += v[i] * c
        end
    end
    return a
end

_adddiag(a::Union{Matrix, SparseMatrixCSC}, v::Real) = _adddiag!(copy(a), v)
_adddiag(a::Union{Matrix, SparseMatrixCSC}, v::AbstractVector, c::Real) = _adddiag!(copy(a), v, c)
_adddiag(a::Union{Matrix, SparseMatrixCSC}, v::AbstractVector{T}) where {T<:Real} = _adddiag!(copy(a), v, one(T))


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
            @inbounds v += a[i, j]*b[i, j]
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
        r[j] = v*c
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

# `rdiv!(::AbstractArray, ::Number)` was introduced in Julia 1.2
# https://github.com/JuliaLang/julia/pull/31179
@static if VERSION < v"1.2.0-DEV.385"
    function _rdiv!(X::AbstractArray, s::Number)
        @simd for I in eachindex(X)
            @inbounds X[I] /= s
        end
        X
    end
else
    _rdiv!(X::AbstractArray, s::Number) = rdiv!(X, s)
end

# `ldiv!(::AbstractArray, ::Number, ::AbstractArray)` was introduced in Julia 1.4
# https://github.com/JuliaLang/julia/pull/33806
@static if VERSION < v"1.4.0-DEV.635"
    _ldiv!(Y::AbstractArray, s::Number, X::AbstractArray) = Y .= s .\ X
else
    _ldiv!(Y::AbstractArray, s::Number, X::AbstractArray) = ldiv!(Y, s, X)
end

# https://github.com/JuliaLang/julia/pull/29749
if VERSION < v"1.1.0-DEV.792"
    eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))
end
