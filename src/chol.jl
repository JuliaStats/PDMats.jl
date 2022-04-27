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

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf.L
end
