"""
Sparse positive definite matrix together with a Cholesky factorization object.
"""
const PDSparseMat{T<:Real,S<:AbstractSparseMatrix,C<:CholTypeSparse} = PDMat{T,S,C}

function PDMats.PDMat(mat::AbstractSparseMatrix, chol::CholTypeSparse)
    d = size(mat, 1)
    size(chol, 1) == d ||
      throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{eltype(mat),typeof(mat),typeof(chol)}(d, mat, chol)
end

PDMats.PDMat(mat::SparseMatrixCSC) = PDMat(mat, cholesky(mat))
PDMats.PDMat(fac::CholTypeSparse) = PDMat(sparse(fac), fac)

PDMats.AbstractPDMat(A::CholTypeSparse) = PDMat(A)

### Conversion
function Base.convert(::Type{PDMat{T}}, a::PDSparseMat) where {T<:Real}
    # CholTypeSparse only supports Float64 and ComplexF64 type parameters!
    # So there is no point in recomputing `cholesky(mat)` and we just reuse
    # the existing Cholesky factorization
    mat = convert(AbstractMatrix{T}, a.mat)
    return PDMat{T,typeof(mat),typeof(a.chol)}(a.dim, mat, a.chol)
end

### Arithmetics

Base.:\(a::PDSparseMat{T}, x::AbstractVecOrMat{T}) where {T<:Real} = convert(Array{T},a.chol \ convert(Array{Float64},x)) #to avoid limitations in sparse factorization library CHOLMOD, see e.g., julia issue #14076
Base.:/(x::AbstractVecOrMat{T}, a::PDSparseMat{T}) where {T<:Real} = convert(Array{T},convert(Array{Float64},x) / a.chol )

### Algebra

Base.inv(a::PDSparseMat{T}) where {T<:Real} = PDMat(inv(a.mat))
LinearAlgebra.det(a::PDSparseMat) = det(a.chol)
LinearAlgebra.logdet(a::PDSparseMat) = logdet(a.chol)
LinearAlgebra.sqrt(A::PDSparseMat) = PDMat(sqrt(Hermitian(Matrix(A))))

### whiten and unwhiten

function PDMats.whiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    # Can't use `ldiv!` due to missing support in SparseArrays
    return copyto!(r, PDMats.chol_lower(a.chol) \ x)
end

function PDMats.unwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(a.chol) * x`
    C = a.chol
    PtL = sparse(C.L)[C.p, :]
    # Can't use `lmul!` due to missing support in SparseArrays
    return copyto!(r, PtL * x)
end


### quadratic forms

PDMats.quad(a::PDSparseMat, x::AbstractVector) = dot(x, a * x)
PDMats.invquad(a::PDSparseMat, x::AbstractVector) = dot(x, a \ x)

function PDMats.quad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    PDMats.@check_argdims eachindex(r) == axes(x, 2)
    for i in axes(x, 2)
        r[i] = quad(a, x[:,i])
    end
    return r
end

function PDMats.invquad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    PDMats.@check_argdims eachindex(r) == axes(x, 2)
    for i in axes(x, 2)
        r[i] = invquad(a, x[:,i])
    end
    return r
end


### tri products

function PDMats.X_A_Xt(a::PDSparseMat, x::AbstractMatrix)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `x * chol_lower(a.chol)`
    C = a.chol
    PtL = sparse(C.L)[C.p, :]
    z = x * PtL
    z * transpose(z)
end


function PDMats.Xt_A_X(a::PDSparseMat, x::AbstractMatrix)
    # `*` is not defined for `UP` factor components,
    # so we can't use `chol_upper(a.chol) * x`
    # Moreover, `sparse` is only defined for `L` factor components
    C = a.chol
    UP = transpose(sparse(C.L))[:, C.p]
    z = UP * x
    transpose(z) * z
end


function PDMats.X_invA_Xt(a::PDSparseMat, x::AbstractMatrix)
    z = a.chol \ collect(transpose(x))
    x * z
end

function PDMats.Xt_invA_X(a::PDSparseMat, x::AbstractMatrix)
    z = a.chol \ x
    transpose(x) * z
end
