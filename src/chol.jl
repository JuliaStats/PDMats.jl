# needs special attention to Cholesky, as the syntax & behavior changes in 0.4-pre

if VERSION >= v"0.4.0-dev+4370"
    # error("Choleksy changes in 0.4.0-dev+4370 (PR #10862), we are still working to make it work with these changes.")

    typealias CholType Cholesky{Float64, Matrix{Float64}}
    chol_lower(a::Matrix{Float64}) = chol(a, Val{:L})

    typealias CholTypeSparse Base.SparseMatrix.CHOLMOD.Factor{Float64}

    chol_lower(cf::CholTypeSparse) = cf[:L]
else
    typealias CholType Cholesky{Float64}
    chol_lower(a::Matrix{Float64}) = chol(a, :L)

    typealias CholTypeSparse Base.LinAlg.CHOLMOD.CholmodFactor{Float64}

    chol_lower(cf::CholTypeSparse) = cf
end
