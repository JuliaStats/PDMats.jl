# Sparse positive definite matrix together with a Cholesky factorization object
immutable PDSparseMat <: AbstractPDMat
    dim::Int
    mat::SparseMatrixCSC{Float64}
    chol::CholmodFactor{Float64}
end

function PDSparseMat(mat::SparseMatrixCSC{Float64}, chol::CholmodFactor{Float64})
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDSparseMat(d, mat, chol)
end

PDSparseMat(mat::SparseMatrixCSC{Float64}) = PDSparseMat(mat, cholfact(mat))

PDSparseMat(fac::CholmodFactor{Float64}) = PDSparseMat(size(fac,1), sparse(fac) |> x -> x*x', fac)

### Basics

dim(a::PDSparseMat) = a.dim
full(a::PDSparseMat) = full(a.mat)
diag(a::PDSparseMat) = diag(a.mat)


### Arithmetics

# add `a * c` to a dense matrix `m` of the same size inplace.
function pdadd!(r::Matrix{Float64}, a::Matrix{Float64}, b::PDSparseMat, c::Real)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, convert(Float64, c))
end

* (a::PDSparseMat, c::Float64) = PDSparseMat(a.mat * c)
* (a::PDSparseMat, x::DenseVecOrMat) = a.mat * x
\ (a::PDSparseMat, x::DenseVecOrMat) = a.chol \ x


### Algebra

inv(a::PDSparseMat) = PDMat( a\eye(a.dim) )
logdet(a::PDSparseMat) = logdet(a.chol)
eigmax(a::PDSparseMat) = eigs(a.mat, which=:LM, nev=1, ritzvec=false)[1][1]
eigmin(a::PDSparseMat) = eigs(a.mat, which=:SM, nev=1, ritzvec=false)[1][1]


### whiten and unwhiten

function whiten!(r::DenseVecOrMat{Float64}, a::PDSparseMat, x::DenseVecOrMat{Float64})
    r[:] = sparse(a.chol) \ x
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
    return r
end

function invquad!(r::AbstractArray, a::PDSparseMat, x::DenseMatrix{Float64})
    for i in 1:size(x, 2)
        r[i] = invquad(a, x[:,i])
    end
    return r
end


### tri products

function X_A_Xt(a::PDSparseMat, x::DenseMatrix{Float64})
    z = x*sparse(a.chol)
    A_mul_Bt(z, z)
end


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
