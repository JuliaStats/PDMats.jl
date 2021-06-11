CholType{T,S<:AbstractMatrix} = Cholesky{T,S}
chol_lower(a::Matrix) = cholesky(a).L

# always use the stored cholesky factor, not a copy
@inline chol_lower(a::CholType) = a.uplo === 'L' ? a.L : transpose(a.U)
@inline chol_upper(a::CholType) = a.uplo === 'U' ? a.U : transpose(a.L)

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf.L
end
