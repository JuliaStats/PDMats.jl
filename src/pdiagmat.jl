# positive diagonal matrix

immutable PDiagMat <: AbstractPDMat
    dim::Int
    diag::Vector{Float64}
    inv_diag::Vector{Float64}
    
    PDiagMat(v::Vector{Float64}) = new(length(v), v, 1.0 ./ v)    
    
    function PDiagMat(v::Vector{Float64}, inv_v::Vector{Float64})
        @check_argdims length(v) == length(inv_v)
        new(length(v), v, inv_v)
    end
end

# basics

dim(a::PDiagMat) = a.dim
full(a::PDiagMat) = diagm(a.diag)
inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
logdet(a::PDiagMat) = sum(log(a.diag))
diag(a::PDiagMat) = copy(a.diag)

* (a::PDiagMat, c::Float64) = PDiagMat(a.diag * c)
* (a::PDiagMat, x::Vector{Float64}) = a.diag .* x
\ (a::PDiagMat, x::Vector{Float64}) = a.inv_diag .* x
* (a::PDiagMat, x::Matrix{Float64}) = mulcols(x, a.diag)
\ (a::PDiagMat, x::Matrix{Float64}) = mulcols(x, a.inv_diag)

# whiten and unwhiten 

whiten(a::PDiagMat, x::Vector{Float64}) = mulsqrt(x, a.inv_diag)
whiten(a::PDiagMat, x::Matrix{Float64}) = mulcols(x, sqrt(a.inv_diag))

whiten!(a::PDiagMat, x::Vector{Float64}) = mulsqrt!(x, a.inv_diag)
whiten!(a::PDiagMat, x::Matrix{Float64}) = mulcols!(x, sqrt(a.inv_diag))

unwhiten(a::PDiagMat, x::Vector{Float64}) = mulsqrt(x, a.diag)
unwhiten(a::PDiagMat, x::Matrix{Float64}) = mulcols(x, sqrt(a.diag))

unwhiten!(a::PDiagMat, x::Vector{Float64}) = mulsqrt!(x, a.diag)
unwhiten!(a::PDiagMat, x::Matrix{Float64}) = mulcols!(x, sqrt(a.diag))

unwhiten_winv!(J::PDiagMat, z::StridedVecOrMat{Float64}) = whiten!(J, z)
unwhiten_winv(J::PDiagMat, z::StridedVecOrMat{Float64}) = whiten(J, z)

# quadratic forms

quad(a::PDiagMat, x::Vector{Float64}) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::Vector{Float64}) = wsumsq(a.inv_diag, x)

quad!(r::Array{Float64}, a::PDiagMat, x::Matrix{Float64}) = gemv!('T', 1., x .* x, a.diag, 0., r)
invquad!(r::Array{Float64}, a::PDiagMat, x::Matrix{Float64}) = gemv!('T', 1., x .* x, a.inv_diag, 0., r)

function X_A_Xt(a::PDiagMat, x::Matrix{Float64}) 
    z = mulrows(x, sqrt(a.diag))
    gemm('N', 'T', 1.0, z, z)
end

function Xt_A_X(a::PDiagMat, x::Matrix{Float64})
    z = mulcols(x, sqrt(a.diag))
    gemm('T', 'N', 1.0, z, z)
end

function X_invA_Xt(a::PDiagMat, x::Matrix{Float64})
    z = mulrows(x, sqrt(a.inv_diag))
    gemm('N', 'T', 1.0, z, z)
end

function Xt_invA_X(a::PDiagMat, x::Matrix{Float64})
    z = mulcols(x, sqrt(a.inv_diag))
    gemm('T', 'N', 1.0, z, z)
end

