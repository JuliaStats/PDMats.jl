# Full positive definite matrix together with a Cholesky factorization object
struct PDMat{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::CholType{T,S}
    PDMat{T,S}(d::Int,m::AbstractMatrix{T},c::CholType{T,S}) where {T,S} = new{T,S}(d,m,c)
end

function PDMat(mat::AbstractMatrix,chol::CholType)
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{eltype(mat),typeof(mat)}(d, mat, chol)
end

PDMat(mat::Matrix) = PDMat(mat, cholesky(mat))
PDMat(mat::Symmetric) = PDMat(Matrix(mat))
PDMat(fac::CholType) = PDMat(Matrix(fac), fac)

### Conversion
Base.convert(::Type{PDMat{T}},         a::PDMat) where {T<:Real} = PDMat(convert(AbstractArray{T}, a.mat))
Base.convert(::Type{AbstractArray{T}}, a::PDMat) where {T<:Real} = convert(PDMat{T}, a)

### Basics

dim(a::PDMat) = a.dim
Base.Matrix(a::PDMat) = copy(a.mat)
LinearAlgebra.diag(a::PDMat) = diag(a.mat)


### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::PDMat, c)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*(a::PDMat{S}, c::T) where {S<:Real, T<:Real} = PDMat(a.mat * c)
*(a::PDMat, x::StridedVecOrMat) = a.mat * x
\(a::PDMat, x::StridedVecOrMat) = a.chol \ x


### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.chol))
LinearAlgebra.logdet(a::PDMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PDMat) = eigmax(a.mat)
LinearAlgebra.eigmin(a::PDMat) = eigmin(a.mat)


### whiten and unwhiten

function whiten!(r::StridedVecOrMat, a::PDMat, x::StridedVecOrMat)
    cf = a.chol.UL
    ldiv!(istriu(cf) ? transpose(cf) : cf, _rcopy!(r, x))
end

function unwhiten!(r::StridedVecOrMat, a::PDMat, x::StridedVecOrMat)
    cf = a.chol.UL
    lmul!(istriu(cf) ? transpose(cf) : cf, _rcopy!(r, x))
end


### quadratic forms

quad(a::PDMat, x::StridedVector) = dot(x, a * x)
invquad(a::PDMat, x::StridedVector) = dot(x, a \ x)

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
    z = copy(x)
    cf = a.chol.UL
    rmul!(z, istriu(cf) ? transpose(cf) : cf)
    z * transpose(z)
end

function Xt_A_X(a::PDMat, x::StridedMatrix)
    cf = a.chol.UL
    z = lmul!(istriu(cf) ? cf : transpose(cf), copy(x))
    transpose(z) * z
end

function X_invA_Xt(a::PDMat, x::StridedMatrix)
    cf = a.chol.UL
    z = rdiv!(copy(x), istriu(cf) ? cf : transpose(cf))
    z * transpose(z)
end

function Xt_invA_X(a::PDMat, x::StridedMatrix)
    cf = a.chol.UL
    z = ldiv!(istriu(cf) ? transpose(cf) : cf, copy(x))
    transpose(z) * z
end
