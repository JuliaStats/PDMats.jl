# Useful utilities to support internal implementation


macro check_argdims(cond)
    quote
        ($(cond)) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end

function _addscal!(r::Matrix, a::Matrix, b::Matrix, c::Real)
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


function sumsq{T}(a::AbstractArray{T})
    s = zero(T)
    for i = 1:length(a)
        @inbounds s += abs2(a[i])
    end
    return s
end

function wsumsq(w::AbstractVector, a::AbstractVector)
    @check_argdims(length(a) == length(w))
    s = 0.
    for i = 1:length(a)
        @inbounds s += abs2(a[i]) * w[i]
    end
    return s
end

function mulcols!{T}(r::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractVector{T}) 
    # multiple b to each column of a
    m = size(a, 1)
    n = size(a, 2)
    @check_argdims(size(r) == (m, n) && length(b) == m)
    for j = 1:n
        aj = view(a, :, j)
        rj = view(r, :, j)
        for i = 1:m
            @inbounds rj[i] = aj[i] * b[i]
        end
    end
    r
end

mulcols!{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulcols!(a, a, b)
mulcols{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulcols!(similar(a), a, b)

function mulrows!{T}(r::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractVector{T}) 
    # multiple b to each column of a
    m = size(a, 1)
    n = size(a, 2)
    @check_argdims(size(r) == (m, n) && length(b) == n)
    for j = 1:n
        aj = view(a, :, j)
        rj = view(r, :, j)
        bj = b[j]
        for i = 1:m
            @inbounds rj[i] = aj[i] * bj
        end
    end
    r
end

mulrows!{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulrows!(a, a, b)
mulrows{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulrows!(similar(a), a, b)

function mulsqrt(x::Vector, c::Vector) 
    @check_argdims length(x) == length(c)
    [x[i] * sqrt(c[i]) for i in 1 : length(x)]
end

function mulsqrt!(x::Vector, c::Vector)
    @check_argdims length(x) == length(c)
    for i in 1 : length(x)
        x[i] .*= sqrt(c[i])
    end
    x
end

function add_diag!(a::Matrix, v::Number)
    n = minimum(size(a))::Int
    for i = 1:n
        @inbounds a[i,i] += v
    end
    a
end



add_diag(a::Matrix, v::Number) = add_diag!(copy(a), v)
add_diag(a::Matrix, v::Vector) = add_diag!(copy(a), v)
add_diag(a::Matrix, v::Vector, c::Number) = add_diag!(copy(a), v, c)
