# Accessing a.L directly might involve an extra copy();
# instead, always use the stored Cholesky factor
# Using `a.factors` instead of `a.L` or `a.U` avoids one
# additional `LowerTriangular` or `UpperTriangular` wrapper and
# leads to better performance
function chol_lower(a::Cholesky)
    return a.uplo === 'L' ? LowerTriangular(a.factors) : LowerTriangular(a.factors')
end
function chol_upper(a::Cholesky)
    return a.uplo === 'U' ? UpperTriangular(a.factors) : UpperTriangular(a.factors')
end

# For a dense Matrix, the following allows us to avoid the Adjoint wrapper:
chol_lower(a::Matrix) = cholesky(Symmetric(a, :L)).L
# NOTE: Formally, the line above should use Hermitian() instead of Symmetric(),
# but this currently has an AutoDiff issue in Zygote.jl, and PDMat is
# type-restricted to be Real, so they are equivalent.
chol_upper(a::Matrix) = cholesky(Symmetric(a, :U)).U

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    # Take into account pivoting!
    chol_lower(cf::CholTypeSparse) = cf.PtL
    chol_upper(cf::CholTypeSparse) = cf.UP
end

# Interface for `Cholesky`

dim(A::Cholesky) = LinearAlgebra.checksquare(A)

# whiten
whiten(A::Cholesky, x::AbstractVecOrMat) = chol_lower(A) \ x
whiten!(A::Cholesky, x::AbstractVecOrMat) = ldiv!(chol_lower(A), x)

# unwhiten
unwhiten(A::Cholesky, x::AbstractVecOrMat) = chol_lower(A) * x
unwhiten!(A::Cholesky, x::AbstractVecOrMat) = lmul!(chol_lower(A), x)

# 3-argument whiten/unwhiten
for T in (:AbstractVector, :AbstractMatrix)
    @eval begin
        whiten!(r::$T, A::Cholesky, x::$T) = whiten!(A, copyto!(r, x))
        unwhiten!(r::$T, A::Cholesky, x::$T) = unwhiten!(A, copyto!(r, x))
    end
end

# quad
function quad(A::Cholesky, x::AbstractVector)
    @check_argdims size(A, 1) == length(x)
    return sum(abs2, chol_upper(A) * x)
end
function quad(A::Cholesky, X::AbstractMatrix)
    @check_argdims size(A, 1) == size(X, 1)
    Z = chol_upper(A) * X
    return vec(sum(abs2, Z; dims=1))
end
function quad!(r::AbstractArray, A::Cholesky, X::AbstractMatrix)
    @check_argdims eachindex(r) == axes(X, 2)
    @check_argdims size(A, 1) == size(X, 1)
    aU = chol_upper(A)
    z = similar(r, size(A, 1)) # buffer to save allocations
    @inbounds for i in axes(X, 2)
        copyto!(z, view(X, :, i))
        lmul!(aU, z)
        r[i] = sum(abs2, z)
    end
    return r
end

# invquad
function invquad(A::Cholesky, x::AbstractVector)
    @check_argdims size(A, 1) == size(x, 1)
    return sum(abs2, chol_lower(A) \ x)
end
function invquad(A::Cholesky, X::AbstractMatrix) 
    @check_argdims size(A, 1) == size(X, 1)
    Z = chol_lower(A) \ X
    return vec(sum(abs2, Z; dims=1))
end
function invquad!(r::AbstractArray, A::Cholesky, X::AbstractMatrix)
    @check_argdims eachindex(r) == axes(X, 2)
    @check_argdims size(A, 1) == size(X, 1)
    aL = chol_lower(A)
    z = similar(r, size(A, 1)) # buffer to save allocations
    @inbounds for i in axes(X, 2)
        copyto!(z, view(X, :, i))
        ldiv!(aL, z)
        r[i] = sum(abs2, z)
    end
    return r
end

# tri products

function X_A_Xt(A::Cholesky, X::AbstractMatrix{<:Real})
    @check_argdims size(A, 1) == size(X, 2)
    Z = X * chol_lower(A)
    return Symmetric(Z * transpose(Z))
end

function Xt_A_X(A::Cholesky, X::AbstractMatrix{<:Real})
    @check_argdims size(A, 1) == size(X, 1)
    Z = chol_upper(A) * X
    return Symmetric(transpose(Z) * Z)
end

function X_invA_Xt(A::Cholesky, X::AbstractMatrix{<:Real})
    @check_argdims size(A, 1) == size(X, 2)
    Z = X / chol_upper(A)
    return Symmetric(Z * transpose(Z))
end

function Xt_invA_X(A::Cholesky, X::AbstractMatrix{<:Real})
    @check_argdims size(A, 1) == size(X, 1)
    Z = chol_lower(A) \ X
    return Symmetric(transpose(Z) * Z)
end
