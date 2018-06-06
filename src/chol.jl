CholType{T,S<:AbstractMatrix} = Cholesky{T,S}
chol_lower(a::Matrix) = cholesky(a).L

if HAVE_CHOLMOD
    CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf.L
end
