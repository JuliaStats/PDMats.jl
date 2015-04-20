# Full positive definite matrix together with a Cholesky factorization object

immutable PDMat <: AbstractPDMat
    dim::Int
    mat::Matrix{Float64}
    chol::CholType
end

function PDMat(mat::Matrix{Float64}, chol::Cholesky{Float64})
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat(d, mat, chol)
end

PDMat(mat::Matrix{Float64}) = PDMat(mat, cholfact(mat))
PDMat(mat::Symmetric{Float64}) = PDMat(mat.S)

PDMat(fac::Cholesky) = PDMat(size(fac,1), full(fac), fac)

### Basics

dim(a::PDMat) = a.dim
full(a::PDMat) = copy(a.mat)
diag(a::PDMat) = diag(a.mat)


### Arithmetics

function pdadd!(r::Matrix{Float64}, a::Matrix{Float64}, b::PDMat, c::Real)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, convert(Float64, c))
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
    istriu(cf) ? Ac_mul_B!(cf, _rcopy!(r, x)) : A_mul_B!(cf, _rcopy!(r, x))
    return r
end


### quadratic forms

quad(a::PDMat, x::DenseVector{Float64}) = dot(x, a * x)
invquad(a::PDMat, x::DenseVector{Float64}) = dot(x, a \ x)

quad!(r::AbstractArray, a::PDMat, x::DenseMatrix{Float64}) = colwise_dot!(r, x, a.mat * x)
invquad!(r::AbstractArray, a::PDMat, x::DenseMatrix{Float64}) = colwise_dot!(r, x, a.mat \ x)


### tri products

function X_A_Xt(a::PDMat, x::DenseMatrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_mul_Bc!(z, cf) : A_mul_B!(z, cf)
    A_mul_Bt(z, z)
end

function Xt_A_X(a::PDMat, x::DenseMatrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_mul_B!(cf, z) : Ac_mul_B!(cf, z)
    At_mul_B(z, z)
end

function X_invA_Xt(a::PDMat, x::DenseMatrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_rdiv_B!(z, cf) : A_rdiv_Bc!(z, cf)
    A_mul_Bt(z, z)
end

function Xt_invA_X(a::PDMat, x::DenseMatrix{Float64})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? Ac_ldiv_B!(cf, z) : A_ldiv_B!(cf, z)
    At_mul_B(z, z)
end
