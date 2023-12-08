"""
Full positive definite matrix together with a Cholesky factorization object.
"""
struct PDMat{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    mat::S
    chol::Cholesky{T,S}

    PDMat{T,S}(m::AbstractMatrix{T},c::Cholesky{T,S}) where {T,S} = new{T,S}(m,c)
end

function PDMat(mat::AbstractMatrix,chol::Cholesky{T,S}) where {T,S}
    d = LinearAlgebra.checksquare(mat)
    if size(chol, 1) != d
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    end
    PDMat{T,S}(convert(S, mat), chol)
end

PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))
PDMat(fac::Cholesky) = PDMat(AbstractMatrix(fac), fac)

function Base.getproperty(a::PDMat, s::Symbol)
    if s === :dim
        return size(getfield(a, :mat), 1)
    end
    return getfield(a, s)
end
Base.propertynames(::PDMat) = (:mat, :chol, :dim)

AbstractPDMat(A::Cholesky) = PDMat(A)

### Conversion
Base.convert(::Type{PDMat{T}}, a::PDMat{T}) where {T<:Real} = a
function Base.convert(::Type{PDMat{T}}, a::PDMat) where {T<:Real}
    chol = convert(Cholesky{T}, a.chol)
    S = typeof(chol.factors)
    mat = convert(S, a.mat)
    return PDMat{T,S}(mat, chol)
end
Base.convert(::Type{AbstractPDMat{T}}, a::PDMat) where {T<:Real} = convert(PDMat{T}, a)

### Basics

Base.size(a::PDMat) = (a.dim, a.dim)
Base.Matrix{T}(a::PDMat) where {T} = Matrix{T}(a.mat)
LinearAlgebra.diag(a::PDMat) = diag(a.mat)
LinearAlgebra.cholesky(a::PDMat) = a.chol

### Work with the underlying matrix in broadcasting
Base.broadcastable(a::PDMat) = Base.broadcastable(a.mat)

### Inheriting from AbstractMatrix

Base.IndexStyle(::Type{PDMat{T,S}}) where {T,S} = Base.IndexStyle(S)
# Linear Indexing
Base.@propagate_inbounds Base.getindex(a::PDMat, i::Int) = getindex(a.mat, i)
# Cartesian Indexing
Base.@propagate_inbounds Base.getindex(a::PDMat, I::Vararg{Int, 2}) = getindex(a.mat, I...)

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
    # /(::AbstractVector, ::Cholesky) is not defined
    if VERSION < v"1.9-"
        # return matrix for 1-element vectors `x`, consistent with LinearAlgebra
        return reshape(x, Val(2)) / a.chol
    else
        if x isa AbstractVector
            return vec(reshape(x, Val(2)) / a.chol)
        else
            return x / a.chol
        end
    end
end

### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.chol))
LinearAlgebra.det(a::PDMat) = det(a.chol)
LinearAlgebra.logdet(a::PDMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PDMat) = eigmax(Symmetric(a.mat))
LinearAlgebra.eigmin(a::PDMat) = eigmin(Symmetric(a.mat))
Base.kron(A::PDMat, B::PDMat) = PDMat(kron(A.mat, B.mat), Cholesky(kron(A.chol.U, B.chol.U), 'U', A.chol.info))
LinearAlgebra.sqrt(A::PDMat) = PDMat(sqrt(Hermitian(A.mat)))

### (un)whitening

function whiten!(r::AbstractVecOrMat, a::PDMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    v = _rcopy!(r, x)
    return ldiv!(chol_lower(cholesky(a)), v)
end
function unwhiten!(r::AbstractVecOrMat, a::PDMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    v = _rcopy!(r, x)
    return lmul!(chol_lower(cholesky(a)), v)
end

function whiten(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) \ x
end
function unwhiten(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) * x
end

## quad/invquad

function quad(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    aU_x = chol_upper(cholesky(a)) * x
    if x isa AbstractVector
        return sum(abs2, aU_x)
    else
        return vec(sum(abs2, aU_x; dims = 1))
    end
end

function quad!(r::AbstractArray, a::PDMat, x::AbstractMatrix)
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

function invquad(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    inv_aL_x = chol_lower(cholesky(a)) \ x
    if x isa AbstractVector
        return sum(abs2, inv_aL_x)
    else
        return vec(sum(abs2, inv_aL_x; dims = 1))
    end
end

function invquad!(r::AbstractArray, a::PDMat, x::AbstractMatrix)
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

function X_A_Xt(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = x * chol_lower(a.chol)
    return Symmetric(z * transpose(z))
end

function Xt_A_X(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = chol_upper(a.chol) * x
    return Symmetric(transpose(z) * z)
end

function X_invA_Xt(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = x / chol_upper(a.chol)
    return Symmetric(z * transpose(z))
end

function Xt_invA_X(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = chol_lower(a.chol) \ x
    return Symmetric(transpose(z) * z)
end

### Specializations for `Array` arguments with reduced allocations

function quad(a::PDMat{<:Real,<:Vector}, x::Matrix)
    @check_argdims a.dim == size(x, 1)
    T = typeof(zero(eltype(a)) * abs2(zero(eltype(x))))
    return quad!(Vector{T}(undef, size(x, 2)), a, x)
end

function invquad(a::PDMat{<:Real,<:Vector}, x::Matrix)
    @check_argdims a.dim == size(x, 1)
    T = typeof(abs2(zero(eltype(x))) / zero(eltype(a)))
    return invquad!(Vector{T}(undef, size(x, 2)), a, x)
end

