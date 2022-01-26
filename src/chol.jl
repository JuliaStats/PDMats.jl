# Accessing a.L directly might involve an extra copy();
# instead, always use the stored Cholesky factor:
chol_lower(a::Cholesky) = a.uplo === 'L' ? a.L : a.U'
chol_upper(a::Cholesky) = a.uplo === 'U' ? a.U : a.L'

# For a dense Matrix, the following allows us to avoid the Adjoint wrapper:
chol_lower(a::Matrix) = cholesky(Symmetric(a, :L)).L
# NOTE: Formally, the line above should use Hermitian() instead of Symmetric(),
# but this currently has an AutoDiff issue in Zygote.jl, and PDMat is
# type-restricted to be Real, so they are equivalent.

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf.L
end
