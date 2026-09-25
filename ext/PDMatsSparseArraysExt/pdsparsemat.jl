"""
Sparse positive definite matrix together with a Cholesky factorization object.
"""
const PDSparseMat{T <: Real, S <: AbstractMatrix{T}} = PDMat{T, S, <:CholTypeSparse}

# CHOLMOD supports only a few element types, so only `mat` is converted and `fact` is reused
function PDMats.PDMat{T, S}(mat::AbstractMatrix, fact::CholTypeSparse) where {T <: Real, S <: AbstractMatrix{T}}
    return PDMat{T, S, typeof(fact)}(mat, fact)
end
function PDMats.PDMat{T}(mat::AbstractMatrix, fact::CholTypeSparse) where {T <: Real}
    mat = convert(AbstractMatrix{T}, mat)
    return PDMat{T, typeof(mat), typeof(fact)}(mat, fact)
end
PDMats.PDMat(mat::AbstractMatrix, fact::CholTypeSparse) = PDMat{eltype(mat)}(mat, fact)
PDMats.PDMat(fact::CholTypeSparse) = PDMat(convert(SparseMatrixCSC, sparse(fact)), fact)

PDMats.AbstractPDMat(A::CholTypeSparse) = PDMat(A)

### Arithmetics

# CHOLMOD requires the right-hand side to have the element type of the factorization
function Base.:\(a::PDSparseMat, x::AbstractVecOrMat{<:Real})
    PDMats.@check_argdims a.dim == size(x, 1)
    T = promote_type(eltype(a), eltype(x))
    return convert(Array{T}, a.fact \ convert(Array{eltype(a.fact)}, x))
end
function Base.:/(x::AbstractVecOrMat{<:Real}, a::PDSparseMat)
    PDMats.@check_argdims a.dim == size(x, 2)
    # CHOLMOD does not support `/`, but `a` is symmetric: `x / a == (a \ xᵀ)ᵀ`.
    z = a \ transpose(x)
    return x isa AbstractVector ? vec(z) : permutedims(z)
end

# Only visit the stored entries
function PDMats._rescale(f, a::SparseMatrixCSC, d::AbstractVector)
    PDMats.@check_argdims eachindex(d) == axes(a, 1) == axes(a, 2)
    zd = zero(eltype(d))
    b = similar(a, typeof(f(zero(eltype(a)), zd * zd)))
    rows = rowvals(a)
    vals = nonzeros(a)
    newvals = nonzeros(b)
    for j in axes(a, 2), k in nzrange(a, j)
        newvals[k] = f(vals[k], d[rows[k]] * d[j])
    end
    return b
end

### Algebra

LinearAlgebra.cholesky(a::PDSparseMat) = a.fact
Base.sqrt(A::PDSparseMat) = PDMat(sqrt(Hermitian(Matrix(A))))

### whiten and unwhiten

_PtL(C::CholTypeSparse) = sparse(C.L)[C.p, :]

function PDMats.whiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` due to missing support in SparseArrays
    return copyto!(r, PDMats.chol_lower(cholesky(a)) \ x)
end
function PDMats.invwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    return copyto!(r, _PtL(cholesky(a))' * x)
end
function PDMats.unwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    return copyto!(r, _PtL(cholesky(a)) * x)
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
    return _PtL(cholesky(a))' * x
end
function PDMats.unwhiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    return _PtL(cholesky(a)) * x
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

# Resolve ambiguities with `PDMat` methods that are more specific in the second argument
PDMats.X_A_Xt(a::PDSparseMat, x::ScalMat) = PDMats._congruence(a, x)
PDMats.X_A_Xt(a::PDSparseMat, x::PDiagMat) = PDMats._congruence(a, x)
PDMats.Xt_A_X(a::PDSparseMat, x::ScalMat) = PDMats._congruence(a, x)
PDMats.Xt_A_X(a::PDSparseMat, x::PDiagMat) = PDMats._congruence(a, x)
for f in (:quad, :invquad)
    @eval PDMats.$f(a::PDSparseMat, x::Matrix) = invoke($f, Tuple{PDMat, Matrix}, a, x)
end
