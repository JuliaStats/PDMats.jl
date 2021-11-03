# Accessing a.L directly might involve an extra copy();
# instead, always use the stored Cholesky factor:
chol_lower(a::Cholesky) = a.uplo === 'L' ? a.L : a.U'
chol_upper(a::Cholesky) = a.uplo === 'U' ? a.U : a.L'

chol_lower(a::Matrix) = chol_lower(cholesky(a))

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf.L
end
