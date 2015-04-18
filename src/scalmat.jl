# Scaling matrix

immutable ScalMat <: AbstractPDMat
    dim::Int
    value::Float64
    inv_value::Float64

    ScalMat(d::Int, v::Float64) = new(d, v, 1.0 / v)
    ScalMat(d::Int, v::Float64, inv_v::Float64) = new(d, v, inv_v)
end


### Basics

dim(a::ScalMat) = a.dim
full(a::ScalMat) = diagm(fill(a.value, a.dim))
diag(a::ScalMat) = fill(a.value, a.dim)


### Arithmetics

function pdadd!(r::Matrix{Float64}, a::Matrix{Float64}, b::ScalMat, c::Real)
    @check_argdims size(r) == size(a) == size(b)
    if is(r, a)
        _adddiag!(r, b.value * convert(Float64, c))
    else
        _adddiag!(copy!(r, a), b.value * convert(Float64, c))
    end
    return r
end

* (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value * c)
/ (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value / c)
* (a::ScalMat, x::DenseVecOrMat) = a.value * x
\ (a::ScalMat, x::DenseVecOrMat) = a.inv_value * x


### Algebra

inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
logdet(a::ScalMat) = a.dim * log(a.value)
eigmax(a::ScalMat) = a.value
eigmin(a::ScalMat) = a.value


### whiten and unwhiten

function whiten!(r::DenseVecOrMat{Float64}, a::ScalMat, x::DenseVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    c = sqrt(a.inv_value)
    for i = 1:length(x)
        @inbounds r[i] = x[i] * c
    end
    return r
end

function unwhiten!(r::DenseVecOrMat{Float64}, a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    c = sqrt(a.value)
    for i = 1:length(x)
        @inbounds r[i] = x[i] * c
    end
    return r
end


### quadratic forms

quad(a::ScalMat, x::Vector{Float64}) = sumabs2(x) * a.value
invquad(a::ScalMat, x::Vector{Float64}) = sumabs2(x) * a.inv_value

quad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64}) = colwise_sumsq!(r, x, a.value)
invquad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64}) = colwise_sumsq!(r, x, a.inv_value)


### tri products

function X_A_Xt(a::ScalMat, x::DenseMatrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.value, x, x)
end

function Xt_A_X(a::ScalMat, x::DenseMatrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.value, x, x)
end

function X_invA_Xt(a::ScalMat, x::DenseMatrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.inv_value, x, x)
end

function Xt_invA_X(a::ScalMat, x::DenseMatrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.inv_value, x, x)
end
