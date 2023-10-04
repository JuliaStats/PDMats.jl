const CholTypeSparse{T} = SparseArrays.CHOLMOD.Factor{T}

# Take into account pivoting!
PDMats.chol_lower(cf::CholTypeSparse) = cf.PtL
PDMats.chol_upper(cf::CholTypeSparse) = cf.UP
