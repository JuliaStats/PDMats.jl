"""
Scaling matrix.
"""
struct ScalMat{T<:Real} <: AbstractPDMat{T}
    dim::Int
    value::T
end

### Conversion
Base.convert(::Type{ScalMat{T}}, a::ScalMat) where {T<:Real} = ScalMat(a.dim, T(a.value))
Base.convert(::Type{AbstractArray{T}}, a::ScalMat) where {T<:Real} = convert(ScalMat{T}, a)

### Basics

dim(a::ScalMat) = a.dim
Base.Matrix(a::ScalMat) = Matrix(Diagonal(fill(a.value, a.dim)))
LinearAlgebra.diag(a::ScalMat) = fill(a.value, a.dim)
LinearAlgebra.cholesky(a::ScalMat) = cholesky(Diagonal(fill(a.value, a.dim)))

### Inheriting from AbstractMatrix

function Base.getindex(a::ScalMat, i::Integer)
    ncol, nrow = fldmod1(i, a.dim)
    ncol == nrow ? a.value : zero(eltype(a))
end
Base.getindex(a::ScalMat{T}, i::Integer, j::Integer) where {T} = i == j ? a.value : zero(T)

### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::ScalMat, c)
    @check_argdims size(r) == size(a) == size(b)
    if r === a
        _adddiag!(r, b.value * c)
    else
        _adddiag!(copyto!(r, a), b.value * c)
    end
    return r
end

*(a::ScalMat, c::Real) = ScalMat(a.dim, a.value * c)
/(a::ScalMat, c::Real) = ScalMat(a.dim, a.value / c)
function *(a::ScalMat, x::AbstractVector)
    @check_argdims dim(a) == length(x)
    return a.value * x
end
function *(a::ScalMat, x::AbstractMatrix)
    @check_argdims dim(a) == size(x, 1)
    return a.value * x
end
function \(a::ScalMat, x::AbstractVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    return x / a.value
end
function /(x::AbstractVecOrMat, a::ScalMat)
    @check_argdims dim(a) == size(x, 2)
    # return matrix for 1-element vectors `x`, consistent with LinearAlgebra
    return reshape(x, Val(2)) / a.value
end
Base.kron(A::ScalMat, B::ScalMat) = ScalMat( dim(A) * dim(B), A.value * B.value )

### Algebra

Base.inv(a::ScalMat) = ScalMat(a.dim, inv(a.value))
LinearAlgebra.logdet(a::ScalMat) = a.dim * log(a.value)
LinearAlgebra.eigmax(a::ScalMat) = a.value
LinearAlgebra.eigmin(a::ScalMat) = a.value


### whiten and unwhiten

function whiten!(r::StridedVecOrMat, a::ScalMat, x::StridedVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    _ldiv!(r, sqrt(a.value), x)
end

function unwhiten!(r::StridedVecOrMat, a::ScalMat, x::StridedVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    mul!(r, x, sqrt(a.value))
end


### quadratic forms

quad(a::ScalMat, x::AbstractVector) = sum(abs2, x) * a.value
invquad(a::ScalMat, x::AbstractVector) = sum(abs2, x) / a.value

quad!(r::AbstractArray, a::ScalMat, x::StridedMatrix) = colwise_sumsq!(r, x, a.value)
invquad!(r::AbstractArray, a::ScalMat, x::StridedMatrix) = colwise_sumsqinv!(r, x, a.value)


### tri products

function X_A_Xt(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 2)
    lmul!(a.value, x * transpose(x))
end

function Xt_A_X(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 1)
    lmul!(a.value, transpose(x) * x)
end

function X_invA_Xt(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 2)
    _rdiv!(x * transpose(x), a.value)
end

function Xt_invA_X(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 1)
    _rdiv!(transpose(x) * x, a.value)
end
