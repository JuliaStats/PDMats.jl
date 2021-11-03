CholType{T,S<:AbstractMatrix} = Cholesky{T,S}

# Accessing a.L directly might involve an extra copy();
# instead, always use the stored Cholesky factor:
chol_lower(a::CholType) = a.uplo === 'L' ? a.L : a.U'
chol_upper(a::CholType) = a.uplo === 'U' ? a.U : a.L'

chol_lower(a::Matrix) = chol_lower(cholesky(a))

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf.L
end
