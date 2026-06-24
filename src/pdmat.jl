"""
Full positive definite matrix together with a factorization object.
"""
struct PDMat{T <: Real, S <: AbstractMatrix{T}, F <: Factorization} <: AbstractPDMat{T}
    mat::S
    fact::F

    function PDMat{T, S, F}(mat::S, fact::F) where {T, S <: AbstractMatrix{T}, F <: Factorization}
        d = LinearAlgebra.checksquare(mat)
        if size(fact) != (d, d)
            throw(DimensionMismatch("Dimensions of the matrix and the factorization are inconsistent."))
        end
        # in principle we might want to check that `fact` is a factorization of `mat`,
        # but that's slow
        return new{T, S, F}(mat, fact)
    end
end

# Construction from a matrix and a Cholesky factorization
function PDMat{T, S}(m::AbstractMatrix, c::Cholesky) where {T <: Real, S <: AbstractMatrix{T}}
    c = convert(Cholesky{T, S}, c)
    return PDMat{T, S, typeof(c)}(convert(S, m), c)
end
function PDMat{T}(m::AbstractMatrix, c::Cholesky) where {T <: Real}
    c = convert(Cholesky{T}, c)
    return PDMat{T, typeof(c.factors)}(m, c)
end
# The element type `T` is derived from the matrix so that e.g. `PDMat(::SymTridiagonal{Int}, ::Cholesky)`
# keeps an integer-valued `mat` (xref the `PDMat from SymTridiagonal` test)
PDMat(mat::AbstractMatrix{T}, chol::Cholesky) where {T <: Real} = PDMat{T, typeof(mat), typeof(chol)}(mat, chol)

# Construction from another PDMat
PDMat{T, S}(pdm::PDMat{T, S}) where {T <: Real, S <: AbstractMatrix{T}} = pdm  # since PDMat doesn't support `setindex!` it's not mutable (xref https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#Mutable-collections)
PDMat{T, S}(pdm::PDMat) where {T <: Real, S <: AbstractMatrix{T}} = PDMat{T, S}(pdm.mat, pdm.fact)
PDMat{T}(pdm::PDMat{T}) where {T <: Real} = pdm
PDMat{T}(pdm::PDMat) where {T <: Real} = PDMat{T}(pdm.mat, pdm.fact)
PDMat(pdm::PDMat) = pdm

# Construction from an AbstractMatrix
function PDMat{T, S}(mat::AbstractMatrix) where {T <: Real, S <: AbstractMatrix{T}}
    mat = convert(S, mat)
    return PDMat{T, S}(mat, cholesky(mat))
end
function PDMat{T}(mat::AbstractMatrix) where {T <: Real}
    mat = convert(AbstractMatrix{T}, mat)
    return PDMat{T}(mat, cholesky(mat))
end
PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))

# Construction from a Cholesky factorization
function PDMat{T, S}(c::Cholesky) where {T <: Real, S <: AbstractMatrix{T}}
    c = convert(Cholesky{T, S}, c)
    return PDMat{T, S}(AbstractMatrix(c), c)
end
function PDMat{T}(c::Cholesky) where {T <: Real}
    c = convert(Cholesky{T}, c)
    return PDMat{T}(AbstractMatrix(c), c)
end
PDMat(c::Cholesky) = PDMat(AbstractMatrix(c), c)

function Base.getproperty(a::PDMat, s::Symbol)
    if s === :dim
        return size(getfield(a, :mat), 1)
    end
    return getfield(a, s)
end
Base.propertynames(::PDMat) = (:mat, :fact, :dim)

AbstractPDMat(A::Cholesky) = PDMat(A)

### Type alias for `PDMat`s backed by a (dense) `Cholesky` factorization, used to define
### `cholesky` (which must return a `Cholesky`). Sparse `PDMat`s (backed by a `CHOLMOD.Factor`)
### define their own `cholesky` method in the SparseArrays extension.
const PDMatCholesky{T <: Real, S <: AbstractMatrix{T}} = PDMat{T, S, <:Cholesky}

### Conversion
# This next method isn't needed because PDMat{T}(a) returns `a` directly
# Base.convert(::Type{PDMat{T}}, a::PDMat{T}) where {T<:Real} = a
Base.convert(::Type{PDMat{T}}, a::PDMat) where {T <: Real} = PDMat{T}(a)
Base.convert(::Type{PDMat{T, S}}, a::PDMat) where {T <: Real, S <: AbstractMatrix{T}} = PDMat{T, S}(a)
function Base.convert(::Type{PDMat{T, S, F}}, a::PDMat) where {T <: Real, S <: AbstractMatrix{T}, F <: Factorization}
    return convert(PDMat{T, S}, a)
end

Base.convert(::Type{AbstractPDMat{T}}, a::PDMat) where {T <: Real} = convert(PDMat{T}, a)

### Basics

Base.size(a::PDMat) = (a.dim, a.dim)
Base.Matrix{T}(a::PDMat) where {T} = Matrix{T}(a.mat)
LinearAlgebra.diag(a::PDMat) = diag(a.mat)
LinearAlgebra.cholesky(a::PDMatCholesky) = a.fact

### Work with the underlying matrix in broadcasting
Base.broadcastable(a::PDMat) = Base.broadcastable(a.mat)

### Inheriting from AbstractMatrix

Base.IndexStyle(::Type{<:PDMat{T, S}}) where {T, S} = Base.IndexStyle(S)
# Linear Indexing
Base.@propagate_inbounds Base.getindex(a::PDMat, i::Int) = getindex(a.mat, i)
# Cartesian Indexing
Base.@propagate_inbounds Base.getindex(a::PDMat, i::Int, j::Int) = getindex(a.mat, i, j)

### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::PDMat, c)
    @check_argdims size(r) == size(a) == size(b)
    return _addscal!(r, a, b.mat, c)
end

*(a::PDMat, c::Real) = PDMat(a.mat * c)
*(a::PDMat, x::AbstractVector) = a.mat * x
*(a::PDMat, x::AbstractMatrix) = a.mat * x
\(a::PDMat, x::AbstractVecOrMat) = a.fact \ x
function /(x::AbstractVecOrMat, a::PDMat)
    # /(::AbstractVector, ::Cholesky) is not defined
    if x isa AbstractVector
        return vec(reshape(x, Val(2)) / a.fact)
    else
        return x / a.fact
    end
end

### Algebra

Base.inv(a::PDMat) = PDMat(inv(a.fact))
LinearAlgebra.det(a::PDMat) = det(a.fact)
LinearAlgebra.logdet(a::PDMat) = logdet(a.fact)
LinearAlgebra.eigmax(a::PDMat) = eigmax(Symmetric(a.mat))
LinearAlgebra.eigmin(a::PDMat) = eigmin(Symmetric(a.mat))
function Base.kron(A::PDMat, B::PDMat)
    M = kron(A.mat, B.mat)
    C = Cholesky(UpperTriangular(kron(chol_upper(cholesky(A)), chol_upper(cholesky(B)))))
    return PDMat(M, C)
end
Base.sqrt(A::PDMat) = PDMat(sqrt(Hermitian(A.mat)))

### (un)whitening

function whiten!(r::AbstractVecOrMat, a::PDMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    if r === x
        return ldiv!(chol_lower(cholesky(a)), r)
    else
        return ldiv!(r, chol_lower(cholesky(a)), x)
    end
end
function invwhiten!(r::AbstractVecOrMat, a::PDMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    if r === x
        return lmul!(chol_upper(cholesky(a)), r)
    else
        return mul!(r, chol_upper(cholesky(a)), x)
    end
end
function unwhiten!(r::AbstractVecOrMat, a::PDMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    if r === x
        return lmul!(chol_lower(cholesky(a)), r)
    else
        return mul!(r, chol_lower(cholesky(a)), x)
    end
end
function invunwhiten!(r::AbstractVecOrMat, a::PDMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    if r === x
        return ldiv!(chol_upper(cholesky(a)), r)
    else
        return ldiv!(r, chol_upper(cholesky(a)), x)
    end
end

function whiten(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) \ x
end
function invwhiten(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_upper(cholesky(a)) * x
end
function unwhiten(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) * x
end
function invunwhiten(a::PDMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_upper(cholesky(a)) \ x
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
        mul!(z, aU, view(x, :, i))
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
    z = x * chol_lower(cholesky(a))
    return Symmetric(z * transpose(z))
end

function Xt_A_X(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = chol_upper(cholesky(a)) * x
    return Symmetric(transpose(z) * z)
end

function X_invA_Xt(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = x / chol_upper(cholesky(a))
    return Symmetric(z * transpose(z))
end

function Xt_invA_X(a::PDMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = chol_lower(cholesky(a)) \ x
    return Symmetric(transpose(z) * z)
end

### Specializations for `Array` arguments with reduced allocations

function quad(a::PDMat{T, Matrix{T}}, x::Matrix) where {T <: Real}
    @check_argdims a.dim == size(x, 1)
    S = typeof(zero(T) * abs2(zero(eltype(x))))
    return quad!(Vector{S}(undef, size(x, 2)), a, x)
end

function invquad(a::PDMat{T, Matrix{T}}, x::Matrix) where {T <: Real}
    @check_argdims a.dim == size(x, 1)
    S = typeof(abs2(zero(eltype(x))) / zero(T))
    return invquad!(Vector{S}(undef, size(x, 2)), a, x)
end
