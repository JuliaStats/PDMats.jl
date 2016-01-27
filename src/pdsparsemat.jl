# Sparse positive definite matrix together with a Cholesky factorization object
immutable PDSparseMat{T<:AbstractFloat,S<:AbstractSparseMatrix} <: AbstractPDMat{T}
  dim::Int
  mat::S
  chol::CholTypeSparse
  PDSparseMat(d::Int,m::AbstractSparseMatrix{T},c::CholTypeSparse) = new(d,m,c) #add {T} to CholTypeSparse argument once #14076 is implemented
end

function PDSparseMat(mat::AbstractSparseMatrix,chol::CholTypeSparse)
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDSparseMat{eltype(mat),typeof(mat)}(d, mat, chol)
end

PDSparseMat(mat::SparseMatrixCSC) = PDSparseMat(mat, cholfact(mat))
PDSparseMat(fac::CholTypeSparse) = PDSparseMat(sparse(fac) |> x -> x*x', fac)

### Basics

dim(a::PDSparseMat) = a.dim
full(a::PDSparseMat) = full(a.mat)
diag(a::PDSparseMat) = diag(a.mat)


### Arithmetics

# add `a * c` to a dense matrix `m` of the same size inplace.
function pdadd!{T<:AbstractFloat}(r::Matrix{T}, a::Matrix{T}, b::PDSparseMat{T}, c::T)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*{T<:AbstractFloat}(a::PDSparseMat{T}, c::T) = PDSparseMat(a.mat * c)
*{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseVecOrMat{T}) = a.mat * x
\{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseVecOrMat{T}) = convert(Array{T},a.chol \ convert(Array{Float64},x)) #to avoid limitations in sparse factorization library CHOLMOD, see e.g., julia issue #14076


### Algebra

inv{T<:AbstractFloat}(a::PDSparseMat{T}) = PDMat( a\eye(T,a.dim) )
logdet(a::PDSparseMat) = logdet(a.chol)
eigmax{T<:AbstractFloat}(a::PDSparseMat{T}) = convert(T,eigs(convert(SparseMatrixCSC{Float64,Int},a.mat), which=:LM, nev=1, ritzvec=false)[1][1]) #to avoid type instability issues in eigs, see e.g., julia issue #13929
eigmin{T<:AbstractFloat}(a::PDSparseMat{T}) = convert(T,eigs(convert(SparseMatrixCSC{Float64,Int},a.mat), which=:SM, nev=1, ritzvec=false)[1][1]) #to avoid type instability issues in eigs, see e.g., julia issue #13929


### whiten and unwhiten

function whiten!{T<:AbstractFloat}(r::DenseVecOrMat{T}, a::PDSparseMat{T}, x::DenseVecOrMat{T})
    r[:] = sparse(chol_lower(a.chol)) \ x
    return r
end

function unwhiten!{T<:AbstractFloat}(r::DenseVecOrMat{T}, a::PDSparseMat{T}, x::StridedVecOrMat{T})
    r[:] = sparse(chol_lower(a.chol)) * x
    return r
end


### quadratic forms

quad{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseVector{T}) = dot(x, a * x)
invquad{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseVector{T}) = dot(x, a \ x)

function quad!{T<:AbstractFloat}(r::AbstractArray{T}, a::PDSparseMat{T}, x::DenseMatrix{T})
    for i in 1:size(x, 2)
        r[i] = quad(a, x[:,i])
    end
    return r
end

function invquad!{T<:AbstractFloat}(r::AbstractArray{T}, a::PDSparseMat{T}, x::DenseMatrix{T})
    for i in 1:size(x, 2)
        r[i] = invquad(a, x[:,i])
    end
    return r
end


### tri products

function X_A_Xt{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseMatrix{T})
    z = x*sparse(chol_lower(a.chol))
    A_mul_Bt(z, z)
end


function Xt_A_X{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseMatrix{T})
    z = At_mul_B(x, sparse(chol_lower(a.chol)))
    A_mul_Bt(z, z)
end


function X_invA_Xt{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseMatrix{T})
    z = a \ x'
    x * z
end

function Xt_invA_X{T<:AbstractFloat}(a::PDSparseMat{T}, x::DenseMatrix{T})
    z = a \ x
    At_mul_B(x, z)
end
