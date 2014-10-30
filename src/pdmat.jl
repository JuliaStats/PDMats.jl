# Full positive definite matrix together with a Cholesky factorization object

immutable PDMat <: AbstractPDMat
    dim::Int
    mat::Matrix{Float64}
    chol::Cholesky{Float64}
end

function PDMat(mat::Matrix{Float64})
    d = size(mat, 1)
    (d >= 1 && size(mat, 2) == d) ||
        throw(ArgumentError("mat must be a square matrix."))
    PDMat(d, mat, cholfact(mat))
end
PDMat(fac::Cholesky) = PDMat(size(fac,1), full(fac), fac)
PDMat(mat::Symmetric{Float64}) = PDMat(mat.S)


### Basics

dim(a::PDMat) = a.dim
full(a::PDMat) = copy(a.mat)
diag(a::PDMat) = diag(a.mat)


### Arithmetics

function pdadd!(r::Matrix{Float64}, a::Matrix{Float64}, b::PDMat, c::Real)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, float64(c))
end

* (a::PDMat, c::Float64) = PDMat(a.mat * c)
* (a::PDMat, x::DenseVecOrMat) = a.mat * x
\ (a::PDMat, x::DenseVecOrMat) = a.chol \ x


### Algebra

inv(a::PDMat) = PDMat(inv(a.chol))
logdet(a::PDMat) = logdet(a.chol)
eigmax(a::PDMat) = eigmax(a.mat)
eigmin(a::PDMat) = eigmin(a.mat)


### whiten and unwhiten

function whiten!(r::DenseVecOrMat{Float64}, a::PDMat, x::DenseVecOrMat{Float64})
    cf = a.chol[:UL]
    istriu(cf) ? Ac_ldiv_B!(cf, _rcopy!(r, x)) : A_ldiv_B!(cf, _rcopy!(r, x))
    return r
end

function unwhiten!(r::DenseVecOrMat{Float64}, a::PDMat, x::StridedVecOrMat{Float64})
    cf = a.chol[:UL]
    istriu(cf) ? Ac_mul_B!(cf, _rcopy!(r, x)) : A_mul_B!(r, _rcopy!(r, x))
    return r
end


# quadratic forms

quad(a::PDMat, x::Vector{Float64}) = dot(x, a.mat * x)
invquad(a::PDMat, x::Vector{Float64}) = abs2(norm(whiten(a, x)))
    
function quad!(r::Array{Float64}, a::PDMat, x::Matrix{Float64}) # = dot!(r, x, a.mat * x, 1)
    n = size(x,2)
    @check_argdims(length(r) == n)
    ax = a.mat * x
    for j = 1:n
        r[j] = dot(view(ax, :, j), view(x, :, j))
    end
    r
end

function invquad!(r::Array{Float64}, a::PDMat, x::Matrix{Float64}) # = sumsq!(fill!(r, 0.0), whiten(a, x), 1)
    n = size(x, 2)
    @check_argdims(length(r) == n)
    wx = whiten(a, x)
    for j = 1:n
        r[j] = sumabs2(view(wx, :, j))
    end
    return r
end

function X_A_Xt(a::PDMat, x::Matrix{Float64})
    z = copy(x) # dimension checks will be done in the A_mul_B*! methods
    cf = a.chol[:UL]
    istriu(cf) ? A_mul_Bc!(z, cf) : A_mul_B!(z, cf)
    gemm('N', 'T', 1.0, z, z)
end

function Xt_A_X(a::PDMat, x::Matrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_mul_B!(cf, z) : Ac_mul_B!(cf, z)
    gemm('T', 'N', 1.0, z, z)
end

function X_invA_Xt(a::PDMat, x::Matrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_rdiv_B!(z, cf) : A_rdiv_Bc!(z, cf)
    gemm('N','T', 1.0, z, z)
end

function Xt_invA_X(a::PDMat, x::Matrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? Ac_ldiv_B!(cf, z) : A_ldiv_B!(cf, z)
    gemm('T', 'N', 1.0, z, z)
end
