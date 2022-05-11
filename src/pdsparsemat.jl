"""
Sparse positive definite matrix together with a Cholesky factorization object.
"""
struct PDSparseMat{T<:Real,S<:AbstractSparseMatrix} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::CholTypeSparse

    PDSparseMat{T,S}(d::Int,m::AbstractSparseMatrix{T},c::CholTypeSparse) where {T,S} =
        new{T,S}(d,m,c) #add {T} to CholTypeSparse argument once #14076 is implemented
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
LinearAlgebra.cholesky(a::PDSparseMat) = a.chol

### Inheriting from AbstractMatrix

Base.getindex(a::PDSparseMat,i::Integer) = getindex(a.mat, i)
Base.getindex(a::PDSparseMat,I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)

### Arithmetics

# add `a * c` to a dense matrix `m` of the same size inplace.
function pdadd!(r::Matrix, a::Matrix, b::PDSparseMat, c)
    @check_argdims size(r) == size(a) == size(b)
    _addscal!(r, a, b.mat, c)
end

*(a::PDSparseMat, c::Real) = PDSparseMat(a.mat * c)
*(a::PDSparseMat, x::StridedVecOrMat) = a.mat * x
\(a::PDSparseMat{T}, x::StridedVecOrMat{T}) where {T<:Real} = convert(Array{T},a.chol \ convert(Array{Float64},x)) #to avoid limitations in sparse factorization library CHOLMOD, see e.g., julia issue #14076
/(x::StridedVecOrMat{T}, a::PDSparseMat{T}) where {T<:Real} = convert(Array{T},convert(Array{Float64},x) / a.chol )

### Algebra

Base.inv(a::PDSparseMat{T}) where {T<:Real} = PDMat(inv(a.mat))
LinearAlgebra.det(a::PDSparseMat) = det(a.chol)
LinearAlgebra.logdet(a::PDSparseMat) = logdet(a.chol)
LinearAlgebra.sqrt(A::PDSparseMat) = PDMat(sqrt(Hermitian(Matrix(A))))

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
