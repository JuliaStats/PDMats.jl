"""
Scaling matrix.
"""
struct ScalMat{T<:Real} <: AbstractPDMat{T}
    dim::Int
    value::T
    inv_value::T
end

ScalMat(d::Int,v::Real) = ScalMat{typeof(inv(v))}(d, v, inv(v))

### Conversion
Base.convert(::Type{ScalMat{T}}, a::ScalMat) where {T<:Real} = ScalMat(a.dim, T(a.value), T(a.inv_value))
Base.convert(::Type{AbstractArray{T}}, a::ScalMat) where {T<:Real} = convert(ScalMat{T}, a)

### Basics

dim(a::ScalMat) = a.dim
Base.Matrix(a::ScalMat) = Matrix(Diagonal(fill(a.value, a.dim)))
LinearAlgebra.diag(a::ScalMat) = fill(a.value, a.dim)


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

*(a::ScalMat, c::T) where {T<:Real} = ScalMat(a.dim, a.value * c)
/(a::ScalMat{T}, c::T) where {T<:Real} = ScalMat(a.dim, a.value / c)
*(a::ScalMat, x::StridedVecOrMat) = a.value * x
\(a::ScalMat, x::StridedVecOrMat) = a.inv_value * x


### Algebra

Base.inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
LinearAlgebra.logdet(a::ScalMat) = a.dim * log(a.value)
LinearAlgebra.eigmax(a::ScalMat) = a.value
LinearAlgebra.eigmin(a::ScalMat) = a.value


### whiten and unwhiten

function whiten!(r::StridedVecOrMat, a::ScalMat, x::StridedVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    mul!(r, x, sqrt(a.inv_value))
end

function unwhiten!(r::StridedVecOrMat, a::ScalMat, x::StridedVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    mul!(r, x, sqrt(a.value))
end


### quadratic forms

quad(a::ScalMat, x::AbstractVector) = sum(abs2, x) * a.value
invquad(a::ScalMat, x::AbstractVector) = sum(abs2, x) * a.inv_value

quad!(r::AbstractArray, a::ScalMat, x::StridedMatrix) = colwise_sumsq!(r, x, a.value)
invquad!(r::AbstractArray, a::ScalMat, x::StridedMatrix) = colwise_sumsq!(r, x, a.inv_value)


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
    lmul!(a.inv_value, x * transpose(x))
end

function Xt_invA_X(a::ScalMat, x::StridedMatrix)
    @check_argdims dim(a) == size(x, 1)
    lmul!(a.inv_value, transpose(x) * x)
end
