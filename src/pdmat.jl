"""
Full positive definite matrix together with a Cholesky factorization object.
"""
struct PDMat{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::CholType{T,S}

    PDMat{T,S}(d::Int,m::AbstractMatrix{T},c::CholType{T,S}) where {T,S} = new{T,S}(d,m,c)
end

function PDMat(mat::AbstractMatrix,chol::CholType{T,S}) where {T,S}
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{T,S}(d, convert(S, mat), chol)
end

PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))
PDMat(fac::CholType) = PDMat(Matrix(fac), fac)

### Conversion
Base.convert(::Type{PDMat{T}},         a::PDMat) where {T<:Real} = PDMat(convert(AbstractArray{T}, a.mat))
Base.convert(::Type{AbstractArray{T}}, a::PDMat) where {T<:Real} = convert(PDMat{T}, a)
Base.convert(::Type{AbstractArray{T}}, a::PDMat{T}) where {T<:Real} = a

### Basics

dim(a::PDMat) = a.dim
Base.Matrix(a::PDMat) = copy(a.mat)
LinearAlgebra.diag(a::PDMat) = diag(a.mat)
LinearAlgebra.cholesky(a::PDMat) = a.chol

### Inheriting from AbstractMatrix

Base.getindex(a::PDMat, i::Int) = getindex(a.mat, i)
Base.getindex(a::PDMat, I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)

### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::PDMat, c)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*(a::PDMat{S}, c::T) where {S<:Real, T<:Real} = PDMat(a.mat * c)
*(a::PDMat, x::AbstractVector{T}) where {T} = a.mat * x
*(a::PDMat, x::AbstractMatrix{T}) where {T} = a.mat * x
\(a::PDMat, x::AbstractVecOrMat) = a.chol \ x
/(x::AbstractVecOrMat, a::PDMat) = x / a.chol

### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.chol))
LinearAlgebra.logdet(a::PDMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PDMat) = eigmax(a.mat)
LinearAlgebra.eigmin(a::PDMat) = eigmin(a.mat)
Base.kron(A::PDMat, B::PDMat) = PDMat(kron(A.mat, B.mat), Cholesky(kron(A.chol.U, B.chol.U), 'U', A.chol.info))

### whiten and unwhiten

function whiten!(r::StridedVecOrMat, a::PDMat, x::StridedVecOrMat)
    cf = a.chol.U
    v = _rcopy!(r, x)
    ldiv!(transpose(cf), v)
end

function unwhiten!(r::StridedVecOrMat, a::PDMat, x::StridedVecOrMat)
    cf = a.chol.U
    v = _rcopy!(r, x)
    lmul!(transpose(cf), v)
end


### quadratic forms

quad(a::PDMat, x::AbstractVector) = sum(abs2, a.chol.U * x)
invquad(a::PDMat, x::AbstractVector) = sum(abs2, a.chol.L \ x)

"""
    quad!(r::AbstractArray, a::AbstractPDMat, x::StridedMatrix)

Overwrite `r` with the value of the quadratic form defined by `a` applied columnwise to `x`
"""
quad!(r::AbstractArray, a::PDMat, x::StridedMatrix) = colwise_dot!(r, x, a.mat * x)

"""
    invquad!(r::AbstractArray, a::AbstractPDMat, x::StridedMatrix)

Overwrite `r` with the value of the quadratic form defined by `inv(a)` applied columnwise to `x`
"""
invquad!(r::AbstractArray, a::PDMat, x::StridedMatrix) = colwise_dot!(r, x, a.mat \ x)


### tri products

function X_A_Xt(a::PDMat, x::StridedMatrix)
    cf = a.chol.U
    z = rmul!(copy(x), transpose(cf))
    return z * transpose(z)
end

function Xt_A_X(a::PDMat, x::StridedMatrix)
    cf = a.chol.U
    z = lmul!(cf, copy(x))
    return transpose(z) * z
end

function X_invA_Xt(a::PDMat, x::StridedMatrix)
    cf = a.chol.U
    z = rdiv!(copy(x), cf)
    return z * transpose(z)
end

function Xt_invA_X(a::PDMat, x::StridedMatrix)
    cf = a.chol.U
    z = ldiv!(transpose(cf), copy(x))
    return transpose(z) * z
end
