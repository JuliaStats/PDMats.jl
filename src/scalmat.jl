# Scaling matrix

immutable ScalMat{T<:Real} <: AbstractPDMat{T}
  dim::Int
  value::T
  inv_value::T
end

ScalMat(d::Int,v::Real) = ScalMat{typeof(v)}(d, v, one(v) / v)

### Basics

dim(a::ScalMat) = a.dim
full(a::ScalMat) = diagm(fill(a.value, a.dim))
diag(a::ScalMat) = fill(a.value, a.dim)


### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::ScalMat, c)
    @check_argdims size(r) == size(a) == size(b)
    if is(r, a)
        _adddiag!(r, b.value * c)
    else
        _adddiag!(copy!(r, a), b.value * c)
    end
    return r
end

*{T<:Real}(a::ScalMat, c::T) = ScalMat(a.dim, a.value * c)
/{T<:Real}(a::ScalMat{T}, c::T) = ScalMat(a.dim, a.value / c)
*(a::ScalMat, x::StridedVecOrMat) = a.value * x
\(a::ScalMat, x::StridedVecOrMat) = a.inv_value * x


### Algebra

inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
logdet(a::ScalMat) = a.dim * log(a.value)
eigmax(a::ScalMat) = a.value
eigmin(a::ScalMat) = a.value


### whiten and unwhiten

function whiten!(r::StridedVecOrMat, a::ScalMat, x::StridedVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    c = sqrt(a.inv_value)
    for i = 1:length(x)
        @inbounds r[i] = x[i] * c
    end
    return r
end

function unwhiten!(r::StridedVecOrMat, a::ScalMat, x::StridedVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    c = sqrt(a.value)
    for i = 1:length(x)
        @inbounds r[i] = x[i] * c
    end
    return r
end


### quadratic forms

quad(a::ScalMat, x::Vector) = sumabs2(x) * a.value
invquad(a::ScalMat, x::Vector) = sumabs2(x) * a.inv_value

quad!(r::AbstractArray, a::ScalMat, x::Matrix) = colwise_sumsq!(r, x, a.value)
invquad!(r::AbstractArray, a::ScalMat, x::Matrix) = colwise_sumsq!(r, x, a.inv_value)


### tri products

function X_A_Xt(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.value, x, x)
end

function Xt_A_X(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.value, x, x)
end

function X_invA_Xt(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.inv_value, x, x)
end

function Xt_invA_X(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.inv_value, x, x)
end
