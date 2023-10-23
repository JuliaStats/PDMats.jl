"""
Sparse positive definite matrix together with a Cholesky factorization object.
"""
const PDSparseMat{T<:Real,S<:AbstractSparseMatrix,C<:CholTypeSparse} = PDMat{T,S,C}

function PDMats.PDMat(mat::AbstractSparseMatrix, chol::CholTypeSparse)
    d = LinearAlgebra.checksquare(mat)
    size(chol, 1) == d ||
      throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDMat{eltype(mat),typeof(mat),typeof(chol)}(mat, chol)
end
Base.@deprecate PDMat{T,S}(d::Int, m::AbstractSparseMatrix{T}, c::CholTypeSparse) where {T,S} PDSparseMat{T,S,typeof(c)}(m, c)

PDMats.PDMat(mat::SparseMatrixCSC) = PDMat(mat, cholesky(mat))
PDMats.PDMat(fac::CholTypeSparse) = PDMat(sparse(fac), fac)

PDMats.AbstractPDMat(A::CholTypeSparse) = PDMat(A)

### Conversion
function Base.convert(::Type{PDMat{T}}, a::PDSparseMat) where {T<:Real}
    # CholTypeSparse only supports Float64 and ComplexF64 type parameters!
    # So there is no point in recomputing `cholesky(mat)` and we just reuse
    # the existing Cholesky factorization
    mat = convert(AbstractMatrix{T}, a.mat)
    return PDMat{T,typeof(mat),typeof(a.chol)}(mat, a.chol)
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
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` due to missing support in SparseArrays
    return copyto!(r, PDMats.chol_lower(a.chol) \ x)
end

function PDMats.unwhiten!(r::AbstractVecOrMat, a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims a.dim == size(x, 1)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(a.chol) * x`
    C = a.chol
    PtL = sparse(C.L)[C.p, :]
    return copyto!(r, PtL * x)
end

function PDMats.whiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    return PDMats.chol_lower(cholesky(a)) \ x
end

function PDMats.unwhiten(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    # `*` is not defined for `PtL` factor components,
    # so we can't use `chol_lower(a.chol) * x`
    C = a.chol
    PtL = sparse(C.L)[C.p, :]
    return PtL * x
end

### quadratic forms

function PDMats.quad(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    # https://github.com/JuliaLang/julia/commit/2425ae760fb5151c5c7dd0554e87c5fc9e24de73
    if VERSION < v"1.4.0-DEV.92"
        z = a.mat * x
        return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
    else
        return x isa AbstractVector ? dot(x, a.mat, x) : map(Base.Fix1(quad, a), eachcol(x))
    end
end

function PDMats.quad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    PDMats.@check_argdims eachindex(r) == axes(x, 2)
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

function PDMats.invquad(a::PDSparseMat, x::AbstractVecOrMat)
    PDMats.@check_argdims a.dim == size(x, 1)
    z = a.chol \ x
    return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
end

function PDMats.invquad!(r::AbstractArray, a::PDSparseMat, x::AbstractMatrix)
    PDMats.@check_argdims eachindex(r) == axes(x, 2)
    PDMats.@check_argdims a.dim == size(x, 1)
    # Can't use `ldiv!` with buffer due to missing support in SparseArrays
    @inbounds for i in axes(x, 2)
        xi = view(x, :, i)
        r[i] = dot(xi, a.chol \ xi)
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
    z = a.chol \ collect(transpose(x))
    return Symmetric(x * z)
end

function PDMats.Xt_invA_X(a::PDSparseMat, x::AbstractMatrix{<:Real})
    PDMats.@check_argdims a.dim == size(x, 1)
    z = a.chol \ x
    return Symmetric(transpose(x) * z)
end
