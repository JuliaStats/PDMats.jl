"""
Full positive definite matrix together with a Cholesky factorization object.
"""
struct PDMat{T<:Real,S<:AbstractMatrix,C<:Factorization} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::C

    PDMat{T,S,C}(d::Int, m::AbstractMatrix{T}, c::Factorization) where {T,S,C} = new{T,S,C}(d,m,c)
end

function PDMat(mat::AbstractMatrix,chol::Cholesky{T,S}) where {T,S}
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{T,S,Cholesky{T,S}}(d, convert(S, mat), chol)
end

PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))
PDMat(fac::Cholesky) = PDMat(AbstractMatrix(fac), fac)

AbstractPDMat(A::Cholesky) = PDMat(A)

### Conversion
Base.convert(::Type{PDMat{T}}, a::PDMat{T}) where {T<:Real} = a
function Base.convert(::Type{PDMat{T}}, a::PDMat) where {T<:Real}
    chol = convert(Factorization{float(T)}, a.chol)
    mat = convert(AbstractMatrix{T}, a.mat)
    return PDMat{T,typeof(mat),typeof(chol)}(size(mat, 1), mat, chol)
end
Base.convert(::Type{AbstractPDMat{T}}, a::PDMat) where {T<:Real} = convert(PDMat{T}, a)

### Basics

Base.size(a::PDMat) = (a.dim, a.dim)
Base.Matrix(a::PDMat) = Matrix(a.mat)
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

*(a::PDMat, c::Real) = PDMat(a.mat * c)
*(a::PDMat, x::AbstractVector) = a.mat * x
*(a::PDMat, x::AbstractMatrix) = a.mat * x
\(a::PDMat, x::AbstractVecOrMat) = a.chol \ x
function /(x::AbstractVecOrMat, a::PDMat)
    return x isa AbstractVector ? vec(reshape(x, Val(2)) / a.chol) : x / a.chol
end

### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.chol))
LinearAlgebra.det(a::PDMat) = det(a.chol)
LinearAlgebra.logdet(a::PDMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PDMat) = eigmax(a.mat)
LinearAlgebra.eigmin(a::PDMat) = eigmin(a.mat)
LinearAlgebra.sqrt(A::PDMat) = PDMat(sqrt(Hermitian(A.mat)))

function Base.kron(
    A::PDMat{<:Real,<:AbstractMatrix,<:Cholesky},
    B::PDMat{<:Real,<:AbstractMatrix,<:Cholesky},
)
    return PDMat(kron(A.mat, B.mat), Cholesky(kron(A.chol.U, B.chol.U), 'U', A.chol.info))
end

### tri products

function X_A_Xt(a::PDMat, x::AbstractMatrix)
    @check_argdims a.dim == size(x, 2)
    z = x * chol_lower(a.chol)
    return z * transpose(z)
end

function Xt_A_X(a::PDMat, x::AbstractMatrix)
    @check_argdims a.dim == size(x, 1)
    z = chol_upper(a.chol) * x
    return transpose(z) * z
end

function X_invA_Xt(a::PDMat, x::AbstractMatrix)
    @check_argdims a.dim == size(x, 2)
    z = x / chol_upper(a.chol)
    return z * transpose(z)
end

function Xt_invA_X(a::PDMat, x::AbstractMatrix)
    @check_argdims a.dim == size(x, 1)
    z = chol_lower(a.chol) \ x
    return transpose(z) * z
end
