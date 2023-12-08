"""
Sparse positive definite matrix together with a Cholesky factorization object.
"""
struct PDSparseMat{T<:Real,S<:AbstractSparseMatrix} <: AbstractPDMat{T}
    mat::S
    chol::CholTypeSparse

    PDSparseMat{T,S}(m::AbstractSparseMatrix{T},c::CholTypeSparse) where {T,S} =
        new{T,S}(m,c) #add {T} to CholTypeSparse argument once #14076 is implemented
end
@deprecate PDSparseMat{T,S}(d::Int, m::AbstractSparseMatrix{T}, c::CholTypeSparse) where {T,S} PDSparseMat{T,S}(m, c)

function PDSparseMat(mat::AbstractSparseMatrix,chol::CholTypeSparse)
    d = LinearAlgebra.checksquare(mat)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDSparseMat{eltype(mat),typeof(mat)}(mat, chol)
end

PDSparseMat(mat::SparseMatrixCSC) = PDSparseMat(mat, cholesky(mat))
PDSparseMat(fac::CholTypeSparse) = PDSparseMat(sparse(fac), fac)

function Base.getproperty(a::PDSparseMat, s::Symbol)
    if s === :dim
        return size(getfield(a, :mat), 1)
    end
    return getfield(a, s)
end
Base.propertynames(::PDSparseMat) = (:mat, :chol, :dim)

AbstractPDMat(A::SparseMatrixCSC) = PDSparseMat(A)
AbstractPDMat(A::CholTypeSparse) = PDSparseMat(A)

### Conversion
Base.convert(::Type{PDSparseMat{T}}, a::PDSparseMat{T}) where {T<:Real} = a
function Base.convert(::Type{PDSparseMat{T}}, a::PDSparseMat) where {T<:Real}
    # CholTypeSparse only supports Float64 and ComplexF64 type parameters!
    # So there is no point in recomputing `cholesky(mat)` and we just reuse
    # the existing Cholesky factorization
    mat = convert(AbstractMatrix{T}, a.mat)
    return PDSparseMat{T,typeof(mat)}(mat, a.chol)
end
Base.convert(::Type{AbstractPDMat{T}}, a::PDSparseMat) where {T<:Real} = convert(PDSparseMat{T}, a)

### Basics

Base.size(a::PDSparseMat) = (a.dim, a.dim)
Base.Matrix{T}(a::PDSparseMat) where {T} = Matrix{T}(a.mat)
LinearAlgebra.diag(a::PDSparseMat) = diag(a.mat)
LinearAlgebra.cholesky(a::PDSparseMat) = a.chol

### Inheriting from AbstractMatrix

Base.IndexStyle(::Type{PDSparseMat{T,S}}) where {T,S} = IndexStyle(S)
# Linear Indexing
Base.@propagate_inbounds Base.getindex(a::PDSparseMat, i::Int) = getindex(a.mat, i)
# Cartesian Indexing
Base.@propagate_inbounds Base.getindex(a::PDSparseMat, I::Vararg{Int, 2}) = getindex(a.mat, I...)

### Arithmetics

# add `a * c` to a dense matrix `m` of the same size inplace.
function pdadd!(r::Matrix, a::Matrix, b::PDSparseMat, c)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*(a::PDSparseMat, c::Real) = PDSparseMat(a.mat * c)
*(a::PDSparseMat, x::AbstractMatrix) = a.mat * x  # defining these seperately to avoid
*(a::PDSparseMat, x::AbstractVector) = a.mat * x  # ambiguity errors
\(a::PDSparseMat{T}, x::AbstractVecOrMat{T}) where {T<:Real} = convert(Array{T},a.chol \ convert(Array{Float64},x)) #to avoid limitations in sparse factorization library CHOLMOD, see e.g., julia issue #14076
/(x::AbstractVecOrMat{T}, a::PDSparseMat{T}) where {T<:Real} = convert(Array{T},convert(Array{Float64},x) / a.chol )

### Algebra

Base.inv(a::PDSparseMat{T}) where {T<:Real} = PDMat(inv(a.mat))
LinearAlgebra.det(a::PDSparseMat) = det(a.chol)
LinearAlgebra.logdet(a::PDSparseMat) = logdet(a.chol)
LinearAlgebra.sqrt(A::PDSparseMat) = PDMat(sqrt(Hermitian(Matrix(A))))

### whiten and unwhiten

function whiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` due to missing support in SparseArrays
    return copyto!(r, chol_lower(a.chol) \ x)
end

function unwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(a.chol) * x`
    C = a.chol
    PtL = sparse(C.L)[C.p, :]
    return copyto!(r, PtL * x)
end

function whiten(a::PDSparseMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return chol_lower(cholesky(a)) \ x
end

function unwhiten(a::PDSparseMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(a.chol) * x`
    C = a.chol
    PtL = sparse(C.L)[C.p, :]
    return PtL * x
end

### quadratic forms

function quad(a::PDSparseMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    # https://github.com/JuliaLang/julia/commit/2425ae760fb5151c5c7dd0554e87c5fc9e24de73
    if VERSION < v"1.4.0-DEV.92"
        z = a.mat * x
        return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
    else
        return x isa AbstractVector ? dot(x, a.mat, x) : map(Base.Fix1(quad, a), eachcol(x))
    end
end

function quad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    @check_argdims eachindex(r) == axes(x, 2)
    @inbounds for i in axes(x, 2)
        xi = view(x, :, i)
        # https://github.com/JuliaLang/julia/commit/2425ae760fb5151c5c7dd0554e87c5fc9e24de73
        if VERSION < v"1.4.0-DEV.92"
            # Can't use `lmul!` with buffer due to missing support in SparseArrays
            r[i] = dot(xi, a.mat * xi)
        else
            r[i] = dot(xi, a.mat, xi)
        end
    end
    return r
end

function invquad(a::PDSparseMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    z = a.chol \ x
    return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
end

function invquad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    @check_argdims eachindex(r) == axes(x, 2)
    @check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` with buffer due to missing support in SparseArrays
    @inbounds for i in axes(x, 2)
        xi = view(x, :, i)
        r[i] = dot(xi, a.chol \ xi)
    end
    return r
end


### tri products

function X_A_Xt(a::PDSparseMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = a.mat * transpose(x)
    return Symmetric(x * z)
end


function Xt_A_X(a::PDSparseMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = a.mat * x
    return Symmetric(transpose(x) * z)
end


function X_invA_Xt(a::PDSparseMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = a.chol \ collect(transpose(x))
    return Symmetric(x * z)
end

function Xt_invA_X(a::PDSparseMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = a.chol \ x
    return Symmetric(transpose(x) * z)
end
