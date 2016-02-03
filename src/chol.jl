# needs special attention to Cholesky, as the syntax & behavior changes in 0.4-pre

if VERSION >= v"0.5.0-dev+907"
    typealias CholType{T,S<:AbstractMatrix} Cholesky{T,S}
    chol_lower(a::Matrix) = ctranspose(chol(a))

    typealias CholTypeSparse{T} SparseArrays.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf[:L]
elseif VERSION >= v"0.4.0-dev+4370"
    # error("Choleksy changes in 0.4.0-dev+4370 (PR #10862), we are still working to make it work with these changes.")

    typealias CholType{T,S<:AbstractMatrix} Cholesky{T,S}
    chol_lower(a::Matrix) = chol(a, Val{:L})

    typealias CholTypeSparse{T} SparseArrays.CHOLMOD.Factor{T}

    chol_lower(cf::CholTypeSparse) = cf[:L]
else
    typealias CholType{T,S} Cholesky{T}
    chol_lower(a::Matrix) = chol(a, :L)

    typealias CholTypeSparse{T} Base.LinAlg.CHOLMOD.CholmodFactor{T}

    chol_lower(cf::CholTypeSparse) = cf
end
