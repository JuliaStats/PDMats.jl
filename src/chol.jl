@compat CholType{T,S<:AbstractMatrix} = Cholesky{T,S}
chol_lower(a::Matrix) = ctranspose(chol(a))

if isdefined(Base.SparseArrays, :CHOLMOD)
    @compat CholTypeSparse{T} = SparseArrays.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf[:L]
end
