# Scaling matrix

immutable ScalMat <: AbstractPDMat
    dim::Int
    value::Float64
    inv_value::Float64
    
    ScalMat(d::Int, v::Float64) = new(d, v, 1.0 / v)
    ScalMat(d::Int, v::Float64, inv_v::Float64) = new(d, v, inv_v)
end

# basics

dim(a::ScalMat) = a.dim
full(a::ScalMat) = diagm(fill(a.value, a.dim))
inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
logdet(a::ScalMat) = a.dim * log(a.value)
diag(a::ScalMat) = fill(a.value, a.dim)
eigmax(a::ScalMat) = a.value
eigmin(a::ScalMat) = a.value

* (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value * c)
/ (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value / c)
* (a::ScalMat, x::StridedVecOrMat) = a.value * x
\ (a::ScalMat, x::StridedVecOrMat) = a.inv_value * x

# whiten and unwhiten 

function whiten(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    x * sqrt(a.inv_value)
end

function whiten!(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    sv = sqrt(a.inv_value)
    for i = 1:length(x)
        @inbounds x[i] *= sv
    end
    x
end

function unwhiten(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    x * sqrt(a.value)
end

function unwhiten!(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    sv = sqrt(a.value)
    for i = 1:length(x)
        @inbounds x[i] *= sv
    end
    x
end

unwhiten_winv!(J::ScalMat,  z::StridedVecOrMat{Float64}) = whiten!(J, z)
unwhiten_winv(J::ScalMat, z::StridedVecOrMat{Float64}) = whiten(J, z)

# quadratic forms

function quad(a::ScalMat, x::Vector{Float64})
    @check_argdims dim(a) == size(x, 1)
    abs2(nrm2(x)) * a.value
end

function invquad(a::ScalMat, x::Vector{Float64})
    @check_argdims dim(a) == size(x, 1)
    abs2(nrm2(x)) * a.inv_value
end

function quad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims dim(a) == m && length(r) == n
    for j = 1:n
        r[j] = sumsq(view(x, :, j)) * a.value
    end
    r
end

function invquad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims dim(a) == m && length(r) == n
    for j = 1:n
        r[j] = sumsq(view(x, :, j)) * a.inv_value
    end
    r
end

function X_A_Xt(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.value, x, x)
end

function Xt_A_X(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.value, x, x)
end

function X_invA_Xt(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.inv_value, x, x)
end

function Xt_invA_X(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.inv_value, x, x)
end

