"""
Full positive definite matrix together with a factorization object.
"""
struct PDMat{T<:Real,S<:AbstractMatrix{T},F<:Factorization} <: AbstractPDMat{T}
    mat::S
    fact::F

    function PDMat{T,S,F}(mat::S, fact::F) where {T,S<:AbstractMatrix{T},F<:Factorization}
        d = LinearAlgebra.checksquare(mat)
        if size(fact) != (d, d)
            throw(DimensionMismatch("Dimensions of the matrix and the factorization are inconsistent."))
        end
        new{T,S,F}(mat, fact)
    end
end

function PDMat(mat::AbstractMatrix{T}, chol::Cholesky) where {T<:Real}
    PDMat{T,typeof(mat),typeof(chol)}(mat, chol)
end

PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))
PDMat(fac::Cholesky) = PDMat(AbstractMatrix(fac), fac)

function Base.getproperty(a::PDMat, s::Symbol)
    if s === :dim
        return size(getfield(a, :mat), 1)
    end
    return getfield(a, s)
end
Base.propertynames(::PDMat) = (:mat, :fact, :dim)

AbstractPDMat(A::Cholesky) = PDMat(A)

### Abstract type alias
const PDMatCholesky{T<:Real,S<:AbstractMatrix{T}} = PDMat{T,S,<:Cholesky}

### Conversion
Base.convert(::Type{PDMat{T}}, a::PDMat{T}) where {T<:Real} = a
function Base.convert(::Type{PDMat{T}}, a::PDMat) where {T<:Real}
    chol = convert(Factorization{float(T)}, a.fact)
    mat = convert(AbstractMatrix{T}, a.mat)
    return PDMat{T,typeof(mat),typeof(chol)}(mat, chol)
end
Base.convert(::Type{AbstractPDMat{T}}, a::PDMat) where {T<:Real} = convert(PDMat{T}, a)

### Basics

Base.size(a::PDMat) = (a.dim, a.dim)
Base.Matrix(a::PDMat) = Matrix(a.mat)
LinearAlgebra.diag(a::PDMat) = diag(a.mat)
LinearAlgebra.cholesky(a::PDMatCholesky) = a.chol

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
\(a::PDMat, x::AbstractVecOrMat) = a.fact \ x
function /(x::AbstractVecOrMat, a::PDMat)
    # /(::AbstractVector, ::Cholesky) is not defined
    if VERSION < v"1.9-"
        # return matrix for 1-element vectors `x`, consistent with LinearAlgebra
        return reshape(x, Val(2)) / a.fact
    else
        return x isa AbstractVector ? vec(reshape(x, Val(2)) / a.fact) : x / a.fact
    end
end

### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.fact))
LinearAlgebra.det(a::PDMat) = det(a.fact)
LinearAlgebra.logdet(a::PDMat) = logdet(a.fact)
LinearAlgebra.eigmax(a::PDMat) = eigmax(a.mat)
LinearAlgebra.eigmin(a::PDMat) = eigmin(a.mat)
LinearAlgebra.sqrt(A::PDMat) = PDMat(sqrt(Hermitian(A.mat)))

function Base.kron(A::PDMatCholesky, B::PDMatCholesky)
    return PDMat(kron(A.mat, B.mat), Cholesky(kron(A.fact.U, B.fact.U), 'U', A.fact.info))
end

### (un)whitening

function whiten!(r::AbstractVecOrMat, a::PDMatCholesky, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    v = _rcopy!(r, x)
    return ldiv!(chol_lower(cholesky(a)), v)
end
function unwhiten!(r::AbstractVecOrMat, a::PDMatCholesky, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    v = _rcopy!(r, x)
    return lmul!(chol_lower(cholesky(a)), v)
end

function whiten(a::PDMatCholesky, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) \ x
end
function unwhiten(a::PDMatCholesky, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) * x
end

## quad/invquad

function quad(a::PDMatCholesky, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    aU_x = chol_upper(cholesky(a)) * x
    if x isa AbstractVector
        return sum(abs2, aU_x)
    else
        return vec(sum(abs2, aU_x; dims = 1))
    end
end

function quad!(r::AbstractArray, a::PDMatCholesky, x::AbstractMatrix)
    @check_argdims eachindex(r) == axes(x, 2)
    @check_argdims a.dim == size(x, 1)
    aU = chol_upper(cholesky(a))
    z = similar(r, a.dim) # buffer to save allocations
    @inbounds for i in axes(x, 2)
        copyto!(z, view(x, :, i))
        lmul!(aU, z)
        r[i] = sum(abs2, z)
    end
    return r
end

function invquad(a::PDMatCholesky, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    inv_aL_x = chol_lower(cholesky(a)) \ x
    if x isa AbstractVector
        return sum(abs2, inv_aL_x)
    else
        return vec(sum(abs2, inv_aL_x; dims = 1))
    end
end

function invquad!(r::AbstractArray, a::PDMatCholesky, x::AbstractMatrix)
    @check_argdims eachindex(r) == axes(x, 2)
    @check_argdims a.dim == size(x, 1)
    aL = chol_lower(cholesky(a))
    z = similar(r, a.dim) # buffer to save allocations
    @inbounds for i in axes(x, 2)
        copyto!(z, view(x, :, i))
        ldiv!(aL, z)
        r[i] = sum(abs2, z)
    end
    return r
end

### tri products

function X_A_Xt(a::PDMatCholesky, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = x * chol_lower(cholesky(a))
    return Symmetric(z * transpose(z))
end

function Xt_A_X(a::PDMatCholesky, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = chol_upper(cholesky(a)) * x
    return Symmetric(transpose(z) * z)
end

function X_invA_Xt(a::PDMatCholesky, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = x / chol_upper(cholesky(a))
    return Symmetric(z * transpose(z))
end

function Xt_invA_X(a::PDMatCholesky, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = chol_lower(cholesky(a)) \ x
    return Symmetric(transpose(z) * z)
end

### Specializations for `Array` arguments with reduced allocations

function quad(a::PDMatCholesky{T,Matrix{T}}, x::Matrix) where {T<:Real}
    @check_argdims a.dim == size(x, 1)
    S = typeof(zero(T) * abs2(zero(eltype(x))))
    return quad!(Vector{S}(undef, size(x, 2)), a, x)
end

function invquad(a::PDMatCholesky{T,Matrix{T}}, x::Matrix) where {T<:Real}
    @check_argdims a.dim == size(x, 1)
    S = typeof(abs2(zero(eltype(x))) / zero(T))
    return invquad!(Vector{S}(undef, size(x, 2)), a, x)
end

