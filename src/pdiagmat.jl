# positive diagonal matrix

immutable PDiagMat <: AbstractPDMat
    dim::Int
    diag::Vector{Float64}
    inv_diag::Vector{Float64}

    PDiagMat(v::Vector{Float64}) = new(length(v), v, 1.0 ./ v)

    function PDiagMat(v::Vector{Float64}, inv_v::Vector{Float64})
        @check_argdims length(v) == length(inv_v)
        new(length(v), v, inv_v)
    end
end


### Basics

dim(a::PDiagMat) = a.dim
full(a::PDiagMat) = diagm(a.diag)
diag(a::PDiagMat) = copy(a.diag)


### Arithmetics

function pdadd!(r::Matrix{Float64}, a::Matrix{Float64}, b::PDiagMat, c::Real)
    @check_argdims size(r) == size(a) == size(b)
    if is(r, a)
        _adddiag!(r, b.diag, convert(Float64, c))
    else
        _adddiag!(copy!(r, a), b.diag, convert(Float64, c))
    end
    return r
end

* (a::PDiagMat, c::Float64) = PDiagMat(a.diag * c)
* (a::PDiagMat, x::DenseVecOrMat) = a.diag .* x
\ (a::PDiagMat, x::DenseVecOrMat) = a.inv_diag .* x


### Algebra

inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
logdet(a::PDiagMat) = sum(log(a.diag))
eigmax(a::PDiagMat) = maximum(a.diag)
eigmin(a::PDiagMat) = minimum(a.diag)


### whiten and unwhiten

function whiten!(r::DenseVector{Float64}, a::PDiagMat, x::DenseVector{Float64})
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.inv_diag
    for i = 1:n
        r[i] = x[i] * sqrt(v[i])
    end
    return r
end

function unwhiten!(r::DenseVector{Float64}, a::PDiagMat, x::DenseVector{Float64})
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.diag
    for i = 1:n
        r[i] = x[i] * sqrt(v[i])
    end
    return r
end

whiten!(r::DenseMatrix{Float64}, a::PDiagMat, x::DenseMatrix{Float64}) =
    broadcast!(*, r, x, sqrt(a.inv_diag))

unwhiten!(r::DenseMatrix{Float64}, a::PDiagMat, x::DenseMatrix{Float64}) =
    broadcast!(*, r, x, sqrt(a.diag))


### quadratic forms

quad(a::PDiagMat, x::Vector{Float64}) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::Vector{Float64}) = wsumsq(a.inv_diag, x)

quad!(r::AbstractArray, a::PDiagMat, x::Matrix{Float64}) = At_mul_B!(r, abs2(x), a.diag)
invquad!(r::AbstractArray, a::PDiagMat, x::Matrix{Float64}) = At_mul_B!(r, abs2(x), a.inv_diag)


### tri products

function X_A_Xt(a::PDiagMat, x::DenseMatrix{Float64})
    z = x .* reshape(sqrt(a.diag), 1, dim(a))
    A_mul_Bt(z, z)
end

function Xt_A_X(a::PDiagMat, x::DenseMatrix{Float64})
    z = x .* sqrt(a.diag)
    At_mul_B(z, z)
end

function X_invA_Xt(a::PDiagMat, x::DenseMatrix{Float64})
    z = x .* reshape(sqrt(a.inv_diag), 1, dim(a))
    A_mul_Bt(z, z)
end

function Xt_invA_X(a::PDiagMat, x::DenseMatrix{Float64})
    z = x .* sqrt(a.inv_diag)
    At_mul_B(z, z)
end
