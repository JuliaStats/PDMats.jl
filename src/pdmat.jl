# Full positive definite matrix together with a Cholesky factorization object
immutable PDMat{T<:AbstractFloat,S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::CholType{T,S}
    PDMat(d::Int,m::AbstractMatrix{T},c::CholType{T,S}) = new(d,m,c)
end

function PDMat(mat::AbstractMatrix,chol::CholType)
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{eltype(mat),typeof(mat)}(d, mat, chol)
end

PDMat(mat::Matrix) = PDMat(mat,cholfact(mat))
PDMat(mat::Symmetric) = PDMat(full(mat))
PDMat(fac::CholType) = PDMat(full(fac),fac)

### Basics

dim(a::PDMat) = a.dim
full(a::PDMat) = copy(a.mat)
diag(a::PDMat) = diag(a.mat)


### Arithmetics

function pdadd!{T<:AbstractFloat}(r::Matrix{T}, a::Matrix{T}, b::PDMat{T}, c::T)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*{T<:AbstractFloat}(a::PDMat{T}, c::T) = PDMat(a.mat * c)
*{T<:AbstractFloat}(a::PDMat{T}, x::DenseVecOrMat{T}) = a.mat * x
\{T<:AbstractFloat}(a::PDMat{T}, x::DenseVecOrMat{T}) = a.chol \ x


### Algebra

inv(a::PDMat) = PDMat(inv(a.chol))
logdet(a::PDMat) = logdet(a.chol)
eigmax(a::PDMat) = eigmax(a.mat)
eigmin(a::PDMat) = eigmin(a.mat)


### whiten and unwhiten

function whiten!{T<:AbstractFloat}(r::DenseVecOrMat{T}, a::PDMat{T}, x::DenseVecOrMat{T})
    cf = a.chol[:UL]
    istriu(cf) ? Ac_ldiv_B!(cf, _rcopy!(r, x)) : A_ldiv_B!(cf, _rcopy!(r, x))
    return r
end

function unwhiten!{T<:AbstractFloat}(r::DenseVecOrMat{T}, a::PDMat{T}, x::StridedVecOrMat{T})
    cf = a.chol[:UL]
    istriu(cf) ? Ac_mul_B!(cf, _rcopy!(r, x)) : A_mul_B!(cf, _rcopy!(r, x))
    return r
end


### quadratic forms

quad{T<:AbstractFloat}(a::PDMat{T}, x::DenseVector{T}) = dot(x, a * x)
invquad{T<:AbstractFloat}(a::PDMat{T}, x::DenseVector{T}) = dot(x, a \ x)

quad!{T<:AbstractFloat}(r::AbstractArray{T}, a::PDMat{T}, x::DenseMatrix{T}) = colwise_dot!(r, x, a.mat * x)
invquad!{T<:AbstractFloat}(r::AbstractArray{T}, a::PDMat{T}, x::DenseMatrix{T}) = colwise_dot!(r, x, a.mat \ x)


### tri products

function X_A_Xt{T<:AbstractFloat}(a::PDMat{T}, x::DenseMatrix{T})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_mul_Bc!(z, cf) : A_mul_B!(z, cf)
    A_mul_Bt(z, z)
end

function Xt_A_X{T<:AbstractFloat}(a::PDMat{T}, x::DenseMatrix{T})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_mul_B!(cf, z) : Ac_mul_B!(cf, z)
    At_mul_B(z, z)
end

function X_invA_Xt{T<:AbstractFloat}(a::PDMat{T}, x::DenseMatrix{T})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? A_rdiv_B!(z, cf) : A_rdiv_Bc!(z, cf)
    A_mul_Bt(z, z)
end

function Xt_invA_X{T<:AbstractFloat}(a::PDMat{T}, x::DenseMatrix{T})
    z = copy(x)
    cf = a.chol[:UL]
    istriu(cf) ? Ac_ldiv_B!(cf, z) : A_ldiv_B!(cf, z)
    At_mul_B(z, z)
end
