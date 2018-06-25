using Arpack
# Sparse positive definite matrix together with a Cholesky factorization object
struct PDSparseMat{T<:Real,S<:AbstractSparseMatrix} <: AbstractPDMat{T}
  dim::Int
  mat::S
  chol::CholTypeSparse
  PDSparseMat{T,S}(d::Int,m::AbstractSparseMatrix{T},c::CholTypeSparse) where {T,S} = new{T,S}(d,m,c) #add {T} to CholTypeSparse argument once #14076 is implemented
end

function PDSparseMat(mat::AbstractSparseMatrix,chol::CholTypeSparse)
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PDSparseMat{eltype(mat),typeof(mat)}(d, mat, chol)
end

PDSparseMat(mat::SparseMatrixCSC) = PDSparseMat(mat, cholesky(mat))
PDSparseMat(fac::CholTypeSparse) = PDSparseMat(sparse(fac) |> x -> x*x', fac)

### Conversion
Base.convert(::Type{PDSparseMat{T}}, a::PDSparseMat) where {T<:Real} = PDSparseMat(convert(SparseMatrixCSC{T}, a.mat))

### Basics

dim(a::PDSparseMat) = a.dim
Base.Matrix(a::PDSparseMat) = Matrix(a.mat)
LinearAlgebra.diag(a::PDSparseMat) = diag(a.mat)


### Arithmetics

# add `a * c` to a dense matrix `m` of the same size inplace.
function pdadd!(r::Matrix, a::Matrix, b::PDSparseMat, c)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*(a::PDSparseMat, c::T) where {T<:Real} = PDSparseMat(a.mat * c)
*(a::PDSparseMat, x::StridedVecOrMat) = a.mat * x
\(a::PDSparseMat{T}, x::StridedVecOrMat{T}) where {T<:Real} = convert(Array{T},a.chol \ convert(Array{Float64},x)) #to avoid limitations in sparse factorization library CHOLMOD, see e.g., julia issue #14076


### Algebra

Base.inv(a::PDSparseMat{T}) where {T<:Real} = PDMat( a\eye(T,a.dim) )
LinearAlgebra.logdet(a::PDSparseMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PDSparseMat{T}) where {T<:Real} = convert(T,eigs(convert(SparseMatrixCSC{Float64,Int},a.mat), which=:LM, nev=1, ritzvec=false)[1][1]) #to avoid type instability issues in eigs, see e.g., julia issue #13929
LinearAlgebra.eigmin(a::PDSparseMat{T}) where {T<:Real} = convert(T,eigs(convert(SparseMatrixCSC{Float64,Int},a.mat), which=:SM, nev=1, ritzvec=false)[1][1]) #to avoid type instability issues in eigs, see e.g., julia issue #13929


### whiten and unwhiten

function whiten!(r::StridedVecOrMat, a::PDSparseMat, x::StridedVecOrMat)
    r[:] = sparse(chol_lower(a.chol)) \ x
    return r
end

function unwhiten!(r::StridedVecOrMat, a::PDSparseMat, x::StridedVecOrMat)
    r[:] = sparse(chol_lower(a.chol)) * x
    return r
end


### quadratic forms

quad(a::PDSparseMat, x::StridedVector) = dot(x, a * x)
invquad(a::PDSparseMat, x::StridedVector) = dot(x, a \ x)

function quad!(r::AbstractArray, a::PDSparseMat, x::StridedMatrix)
    for i in 1:size(x, 2)
        r[i] = quad(a, x[:,i])
    end
    return r
end

function invquad!(r::AbstractArray, a::PDSparseMat, x::StridedMatrix)
    for i in 1:size(x, 2)
        r[i] = invquad(a, x[:,i])
    end
    return r
end


### tri products

function X_A_Xt(a::PDSparseMat, x::StridedMatrix)
    z = x * sparse(chol_lower(a.chol))
    z * transpose(z)
end


function Xt_A_X(a::PDSparseMat, x::StridedMatrix)
    z = transpose(x) * sparse(chol_lower(a.chol))
    z * transpose(z)
end


function X_invA_Xt(a::PDSparseMat, x::StridedMatrix)
    z = a.chol \ collect(transpose(x))
    x * z
end

function Xt_invA_X(a::PDSparseMat, x::StridedMatrix)
    z = a.chol \ x
    transpose(x) * z
end
