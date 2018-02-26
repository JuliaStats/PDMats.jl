CholType{T,S<:AbstractMatrix} = Cholesky{T,S}
chol_lower(a::Matrix) = chol(a)'

#=
if isdefined(Base.SparseArrays, :CHOLMOD)
    CholTypeSparse{T} = SparseArrays.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf[:L]
end
=#