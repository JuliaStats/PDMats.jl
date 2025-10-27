module StatsBaseExt

using PDMats
using PDMats.LinearAlgebra: Cholesky

using StatsBase: StatsBase, cor2cov, cov2cor

# Fallback
function StatsBase.cor2cov(C::AbstractPDMat, x::AbstractVector{<:Real})
    return X_A_Xt(C, PDiagMat(x))
end

# Exploit possible optimizations of `cor2cov` (for e.g. symmetric matrices)
function StatsBase.cor2cov(C::PDMat, x::AbstractVector{<:Real})
    PDMats.@check_argdims size(C, 1) == length(x)
    mat = cor2cov(C.mat, x)
    uplo = C.chol.uplo
    if uplo === 'U'
        factors = C.chol.factors .* x'
    else
        factors = x .* C.chol.factors
    end
    chol = Cholesky(factors, uplo, C.chol.info)
    return PDMat(mat, chol)
end

# Fallback
function StatsBase.cov2cor(C::AbstractPDMat, x::AbstractVector{<:Real})
    return X_A_Xt(C, PDiagMat(oneunit(eltype(C)) ./ x))
end

# Implementations with reduced allocations
function StatsBase.cov2cor(C::ScalMat, x::AbstractVector{<:Real})
    PDMats.@check_argdims C.dim == length(x)
    return PDiagMat(C.value ./ abs2.(x))
end
StatsBase.cov2cor(C::PDiagMat, x::AbstractVector{<:Real}) = PDiagMat(C.diag ./ abs2.(x))

# Implementations with reduced allocations and exploiting possible optimizations of `cov2cor`
function StatsBase.cov2cor(C::PDMat, x::AbstractVector{<:Real})
    PDMats.@check_argdims size(C, 1) == length(x)
    mat = cov2cor(C.mat, x)
    uplo = C.chol.uplo
    if uplo === 'U'
        factors = C.chol.factors ./ x'
    else
        factors = x .\ C.chol.factors
    end
    chol = Cholesky(factors, uplo, C.chol.info)
    return PDMat(mat, chol)
end

end # module
