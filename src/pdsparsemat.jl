# Sparse positive definite matrix together with a Cholesky factorization object

immutable PDSparseMat <: AbstractPDMat
    dim::Int
    mat::SparseMatrixCSC{Float64, Int64}
    chol::CholmodFactor{Float64, Int64}
end

function PDSparseMat(mat::SparseMatrixCSC{Float64, Int64}, chol::CholmodFactor{Float64, Int64})
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDSparseMat(d, mat, chol)
end

PDSparseMat(mat::SparseMatrixCSC{Float64, Int64}) = PDSparseMat(mat, cholfact(mat))

PDSparseMat(fac::CholmodFactor{Float64, Int64}) = PDSparseMat(size(fac,1), sparse(fac) |> x -> x*x', fac)

### Basics

dim(a::PDSparseMat) = a.dim
full(a::PDSparseMat) = full(a.mat)
diag(a::PDSparseMat) = diag(a.mat)


### Arithmetics

* (a::PDSparseMat, c::Float64) = PDSparseMat(a.mat * c)
* (a::PDSparseMat, x::DenseVecOrMat) = a.mat * x
\ (a::PDSparseMat, x::DenseVecOrMat) = a.chol \ x


### Algebra

inv(a::PDSparseMat) = PDMat( a\eye(a.dim) )
logdet(a::PDSparseMat) = logdet(a.chol)
eigmax(a::PDSparseMat) = maximum(eigs(a.mat, ritzvec = false)[1])
eigmin(a::PDSparseMat) = minimum(eigs(a.mat, ritzvec = false)[1])


### whiten and unwhiten

function whiten!(r::DenseVecOrMat{Float64}, a::PDSparseMat, x::DenseVecOrMat{Float64})
    A_ldiv_B!(a.chol, _rcopy!(r, x))
    return r
end


function unwhiten!(r::DenseVecOrMat{Float64}, a::PDSparseMat, x::StridedVecOrMat{Float64})
    r[:] = sparse(a.chol) * x
    return r
end


### quadratic forms

quad(a::PDSparseMat, x::DenseVector{Float64}) = dot(x, a * x)
invquad(a::PDSparseMat, x::DenseVector{Float64}) = dot(x, a \ x)

function quad!(r::AbstractArray, a::PDSparseMat, x::DenseMatrix{Float64})
    for i in 1:size(x, 2)
        r[i] = quad(a, x[:,i])
    end
end

function invquad!(r::AbstractArray, a::PDSparseMat, x::DenseMatrix{Float64})
    for i in 1:size(x, 2)
        r[i] = invquad(a, x[:,i])
    end
end


### tri products


## function X_A_Xt(a::PDSparseMat, x::DenseMatrix{Float64})
##     z = x*a.mat
##     A_mul_Bt(z, X)
## end

function X_A_Xt(a::PDSparseMat, x::DenseMatrix{Float64})
    z = x*sparse(a.chol)
    A_mul_Bt(z, z)
end

## function Xt_A_X(a::PDSparseMat, x::DenseMatrix{Float64})
##     z = a.mat*x
##     At_mul_B(X, z)
## end

function Xt_A_X(a::PDSparseMat, x::DenseMatrix{Float64})
    z = At_mul_B(x, sparse(a.chol))
    A_mul_Bt(z, z)
end


function X_invA_Xt(a::PDSparseMat, x::DenseMatrix{Float64})
    z = a \ x'
    x * z
end

function Xt_invA_X(a::PDSparseMat, x::DenseMatrix{Float64})
    z = a \ x
    At_mul_B(x, z)
end
