# Scaling matrix

immutable ScalMat{T<:AbstractFloat} <: AbstractPDMat{T}
  dim::Int
  value::T
  inv_value::T
end

ScalMat(d::Int,v::AbstractFloat) = ScalMat{typeof(v)}(d, v, one(v) / v)

### Basics

dim(a::ScalMat) = a.dim
full(a::ScalMat) = diagm(fill(a.value, a.dim))
diag(a::ScalMat) = fill(a.value, a.dim)


### Arithmetics

function pdadd!{T<:AbstractFloat}(r::Matrix{T}, a::Matrix{T}, b::ScalMat{T}, c::T)
    @check_argdims size(r) == size(a) == size(b)
    if is(r, a)
        _adddiag!(r, b.value * c)
    else
        _adddiag!(copy!(r, a), b.value * c)
    end
    return r
end

*{T<:AbstractFloat}(a::ScalMat{T}, c::T) = ScalMat(a.dim, a.value * c)
/{T<:AbstractFloat}(a::ScalMat{T}, c::T) = ScalMat(a.dim, a.value / c)
*{T<:AbstractFloat}(a::ScalMat{T}, x::DenseVecOrMat{T}) = a.value * x
\{T<:AbstractFloat}(a::ScalMat{T}, x::DenseVecOrMat{T}) = a.inv_value * x


### Algebra

inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
logdet(a::ScalMat) = a.dim * log(a.value)
eigmax(a::ScalMat) = a.value
eigmin(a::ScalMat) = a.value


### whiten and unwhiten

function whiten!{T<:AbstractFloat}(r::DenseVecOrMat{T}, a::ScalMat{T}, x::DenseVecOrMat{T})
    @check_argdims dim(a) == size(x, 1)
    c = sqrt(a.inv_value)
    for i = 1:length(x)
        @inbounds r[i] = x[i] * c
    end
    return r
end

function unwhiten!{T<:AbstractFloat}(r::DenseVecOrMat{T}, a::ScalMat{T}, x::StridedVecOrMat{T})
    @check_argdims dim(a) == size(x, 1)
    c = sqrt(a.value)
    for i = 1:length(x)
        @inbounds r[i] = x[i] * c
    end
    return r
end


### quadratic forms

quad{T<:AbstractFloat}(a::ScalMat, x::Vector{T}) = sumabs2(x) * a.value
invquad{T<:AbstractFloat}(a::ScalMat, x::Vector{T}) = sumabs2(x) * a.inv_value

quad!{T<:AbstractFloat}(r::AbstractArray{T}, a::ScalMat{T}, x::Matrix{T}) = colwise_sumsq!(r, x, a.value)
invquad!{T<:AbstractFloat}(r::AbstractArray{T}, a::ScalMat{T}, x::Matrix{T}) = colwise_sumsq!(r, x, a.inv_value)


### tri products

function X_A_Xt{T<:AbstractFloat}(a::ScalMat{T}, x::DenseMatrix{T})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.value, x, x)
end

function Xt_A_X{T<:AbstractFloat}(a::ScalMat{T}, x::DenseMatrix{T})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.value, x, x)
end

function X_invA_Xt{T<:AbstractFloat}(a::ScalMat{T}, x::DenseMatrix{T})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.inv_value, x, x)
end

function Xt_invA_X{T<:AbstractFloat}(a::ScalMat{T}, x::DenseMatrix{T})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.inv_value, x, x)
end
