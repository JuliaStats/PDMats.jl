"""
Full positive definite matrix together with a Cholesky factorization object.
"""
struct PDMat{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    mat::S
    chol::Cholesky{T,S}

    PDMat{T,S}(m::AbstractMatrix{T},c::Cholesky{T,S}) where {T,S} = new{T,S}(m,c)
    function PDMat{T,S}(d::Int, m::AbstractMatrix{T},c::Cholesky{T,S}) where {T,S}
        LinearAlgebra.checksquare(m) == d || throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
        new{T,S}(m,c)
    end
end

function PDMat(mat::AbstractMatrix,chol::Cholesky{T,S}) where {T,S}
    d = LinearAlgebra.checksquare(mat)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{T,S}(convert(S, mat), chol)
end

PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))
PDMat(fac::Cholesky) = PDMat(AbstractMatrix(fac), fac)

function Base.getproperty(a::PDMat, s::Symbol)
    s === :dim && return size(getfield(a, :mat), 1)
    return getfield(a, s)
end
Base.propertynames(::PDMat) = (:mat, :chol, :dim)

AbstractPDMat(A::Cholesky) = PDMat(A)

### Conversion
Base.convert(::Type{PDMat{T}},         a::PDMat) where {T<:Real} = PDMat(convert(AbstractArray{T}, a.mat))
Base.convert(::Type{AbstractArray{T}}, a::PDMat) where {T<:Real} = convert(PDMat{T}, a)
Base.convert(::Type{AbstractArray{T}}, a::PDMat{T}) where {T<:Real} = a

### Basics

Base.size(a::PDMat) = (a.dim, a.dim)
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

*(a::PDMat, c::Real) = PDMat(a.mat * c)
*(a::PDMat, x::AbstractVector) = a.mat * x
*(a::PDMat, x::AbstractMatrix) = a.mat * x
\(a::PDMat, x::AbstractVecOrMat) = a.chol \ x
function /(x::AbstractVecOrMat, a::PDMat)
    if VERSION >= v"1.9" && x isa AbstractVector
        # either size(x) == 1, or we error
        if length(x) != 1 || size(a) != (1,1)
            throw(DimensionMismatch("size of A is $(size(a)), size of B is ($(length(x)), 1)"))
        end
        return x ./ a[1]
    end
    # return matrix for 1-element vectors `x`, consistent with LinearAlgebra
    reshape(x, Val(2)) / a.chol
end
### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.chol))
LinearAlgebra.det(a::PDMat) = det(a.chol)
LinearAlgebra.logdet(a::PDMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PDMat) = eigmax(a.mat)
LinearAlgebra.eigmin(a::PDMat) = eigmin(a.mat)
Base.kron(A::PDMat, B::PDMat) = PDMat(kron(A.mat, B.mat), Cholesky(kron(A.chol.U, B.chol.U), 'U', A.chol.info))
LinearAlgebra.sqrt(A::PDMat) = PDMat(sqrt(Hermitian(A.mat)))

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
