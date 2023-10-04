
if isdefined(Base, :get_extension)
    const CholTypeSparse{T} = SparseArrays.CHOLMOD.Factor{T}
else
    const CholTypeSparse{T} = SuiteSparse.CHOLMOD.Factor{T}
end

# Take into account pivoting!
PDMats.chol_lower(cf::CholTypeSparse) = cf.PtL
PDMats.chol_upper(cf::CholTypeSparse) = cf.UP
