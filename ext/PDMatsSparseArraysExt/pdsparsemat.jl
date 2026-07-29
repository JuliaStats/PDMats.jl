"""
Sparse positive definite matrix together with a Cholesky factorization object.
"""
const PDSparseMat{T <: Real, S <: AbstractSparseMatrix{T}, C <: CholTypeSparse} = PDMat{T, S, C}

function PDMats.PDMat(mat::AbstractSparseMatrix, chol::CholTypeSparse)
    return PDMat{eltype(mat), typeof(mat), typeof(chol)}(mat, chol)
end
Base.@deprecate PDMat{T, S}(d::Int, m::AbstractSparseMatrix{T}, c::CholTypeSparse) where {T, S} PDSparseMat{T, S, typeof(c)}(m, c)

PDMats.PDMat(mat::SparseMatrixCSC) = PDMat(mat, cholesky(mat))
PDMats.PDMat(fac::CholTypeSparse) = PDMat(sparse(fac), fac)

PDMats.AbstractPDMat(A::CholTypeSparse) = PDMat(A)

### Conversion
Base.convert(::Type{PDMat{T}}, a::PDSparseMat{T}) where {T <: Real} = a
function Base.convert(::Type{PDMat{T}}, a::PDSparseMat) where {T <: Real}
    # CholTypeSparse only supports Float64 and ComplexF64 type parameters!
    # So there is no point in recomputing `cholesky(mat)` and we just reuse
    # the existing Cholesky factorization
    mat = convert(AbstractMatrix{T}, a.mat)
    return PDMat{T, typeof(mat), typeof(a.fact)}(mat, a.fact)
end

### Arithmetics

# `x` is converted to `Float64` to work around CHOLMOD limitations (julia issue #14076).
function Base.:\(a::PDSparseMat, x::AbstractVecOrMat{<:Real})
    PDMats.@check_argdims a.dim == size(x, 1)
    T = promote_type(eltype(a), eltype(x))
    return convert(Array{T}, a.fact \ convert(Array{Float64}, x))
end
function Base.:/(x::AbstractVecOrMat{<:Real}, a::PDSparseMat)
    PDMats.@check_argdims a.dim == size(x, 2)
    # CHOLMOD does not support `/`, but `a` is symmetric: `x / a == (a \ xᵀ)ᵀ`.
    z = a \ transpose(x)
    return x isa AbstractVector ? vec(z) : permutedims(z)
end

# `_scaleadddiag`'s single-allocation fast path for mutable storage (xref #239)
PDMats._scaleadddiag(a::SparseMatrixCSC, c::Real, v::Real) = PDMats._adddiag!(a * c, v)

### Algebra

LinearAlgebra.cholesky(a::PDSparseMat) = a.fact
Base.sqrt(A::PDSparseMat) = PDMat(sqrt(Hermitian(Matrix(A))))

### whiten and unwhiten

function PDMats.whiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` due to missing support in SparseArrays
    return copyto!(r, PDMats.chol_lower(cholesky(a)) \ x)
end
function PDMats.invwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # `*` and `mul!` are not defined for `UP` factor components,
    # so we can't use `chol_upper(C) * x`;
    # `sparse` is neither defined for `PtL` nor for `UP` nor for `U` factor components
    C = cholesky(a)
    PtL = sparse(C.L)[C.p, :]
    return copyto!(r, PtL' * x)
end
function PDMats.unwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(C) * x`
    C = cholesky(a)
    PtL = sparse(C.L)[C.p, :]
    return copyto!(r, PtL * x)
end
function PDMats.invunwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` due to missing support in SparseArrays
    return copyto!(r, PDMats.chol_upper(cholesky(a)) \ x)
end

function PDMats.whiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    return PDMats.chol_lower(cholesky(a)) \ x
end
function PDMats.invwhiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    # `*` is not defined for `UP` factor components,
    # so we can't use `chol_upper(C) * x`
    C = cholesky(a)
    PtL = sparse(C.L)[C.p, :]
    return PtL' * x
end
function PDMats.unwhiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(C) * x`
    C = cholesky(a)
    PtL = sparse(C.L)[C.p, :]
    return PtL * x
end
function PDMats.invunwhiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    return PDMats.chol_upper(cholesky(a)) \ x
end

### quadratic forms

function PDMats.quad(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    return x isa AbstractVector ? dot(x, a.mat, x) : map(Base.Fix1(quad, a), eachcol(x))
end

function PDMats.quad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    PDMats.@check_argdims eachindex(r) == axes(x, 2)
    @inbounds for i in axes(x, 2)
        xi = view(x, :, i)
        r[i] = dot(xi, a.mat, xi)
    end
    return r
end

function PDMats.invquad(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    z = cholesky(a) \ x
    return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
end

function PDMats.invquad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    PDMats.@check_argdims eachindex(r) == axes(x, 2)
    PDMats.@check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` with buffer due to missing support in SparseArrays
    C = cholesky(a)
    @inbounds for i in axes(x, 2)
        xi = view(x, :, i)
        r[i] = dot(xi, C \ xi)
    end
    return r
end


### tri products

function PDMats.X_A_Xt(a::PDSparseMat, x::AbstractMatrix{<:Real})
    PDMats.@check_argdims a.dim == size(x, 2)
    z = a.mat * transpose(x)
    return Symmetric(x * z)
end


function PDMats.Xt_A_X(a::PDSparseMat, x::AbstractMatrix{<:Real})
    PDMats.@check_argdims a.dim == size(x, 1)
    z = a.mat * x
    return Symmetric(transpose(x) * z)
end


function PDMats.X_invA_Xt(a::PDSparseMat, x::AbstractMatrix{<:Real})
    PDMats.@check_argdims a.dim == size(x, 2)
    z = cholesky(a) \ collect(transpose(x))
    return Symmetric(x * z)
end

function PDMats.Xt_invA_X(a::PDSparseMat, x::AbstractMatrix{<:Real})
    PDMats.@check_argdims a.dim == size(x, 1)
    z = cholesky(a) \ x
    return Symmetric(transpose(x) * z)
end

# Resolve the ambiguity between the generic `PDMat` methods in `src/congruence.jl`, which are more
# specific in the second argument, and the methods above, which are more specific in the first one
PDMats.X_A_Xt(a::PDSparseMat, x::ScalMat) = PDMats._congruence(a, x)
PDMats.X_A_Xt(a::PDSparseMat, x::PDiagMat) = PDMats._congruence(a, x)
PDMats.Xt_A_X(a::PDSparseMat, x::ScalMat) = PDMats._congruence(a, x)
PDMats.Xt_A_X(a::PDSparseMat, x::PDiagMat) = PDMats._congruence(a, x)
