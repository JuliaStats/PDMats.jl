"""
Scaling matrix.
"""
struct ScalMat{T<:Real} <: AbstractPDMat{T}
    dim::Int
    value::T
end

### Conversion
Base.convert(::Type{ScalMat{T}}, a::ScalMat{T}) where {T<:Real} = a
Base.convert(::Type{ScalMat{T}}, a::ScalMat) where {T<:Real} = ScalMat(a.dim, T(a.value))
Base.convert(::Type{AbstractPDMat{T}}, a::ScalMat) where {T<:Real} = convert(ScalMat{T}, a)

### Basics

Base.size(a::ScalMat) = (a.dim, a.dim)
Base.Matrix(a::ScalMat) = Matrix(Diagonal(fill(a.value, a.dim)))
LinearAlgebra.diag(a::ScalMat) = fill(a.value, a.dim)
LinearAlgebra.cholesky(a::ScalMat) = Cholesky(Diagonal(fill(sqrt(a.value), a.dim)), 'U', 0)

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
    @check_argdims a.dim == length(x)
    return a.value * x
end
function *(a::ScalMat, x::AbstractMatrix)
    @check_argdims a.dim == size(x, 1)
    return a.value * x
end
function \(a::ScalMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return x / a.value
end
function /(x::AbstractVecOrMat, a::ScalMat)
    @check_argdims a.dim == size(x, 2)
    if VERSION < v"1.9-"
        # return matrix for 1-element vectors `x`, consistent with LinearAlgebra < 1.9
        return reshape(x, Val(2)) / a.value
    else
        return x / a.value
    end
end
Base.kron(A::ScalMat, B::ScalMat) = ScalMat(A.dim * B.dim, A.value * B.value )

### Algebra

Base.inv(a::ScalMat) = ScalMat(a.dim, inv(a.value))
LinearAlgebra.det(a::ScalMat) = a.value^a.dim
LinearAlgebra.logdet(a::ScalMat) = a.dim * log(a.value)
LinearAlgebra.eigmax(a::ScalMat) = a.value
LinearAlgebra.eigmin(a::ScalMat) = a.value
LinearAlgebra.sqrt(a::ScalMat) = ScalMat(a.dim, sqrt(a.value))


### whiten and unwhiten

function whiten!(r::AbstractVecOrMat, a::ScalMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    _ldiv!(r, sqrt(a.value), x)
end

function unwhiten!(r::AbstractVecOrMat, a::ScalMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    mul!(r, x, sqrt(a.value))
end

function whiten(a::ScalMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return x / sqrt(a.value)
end
function unwhiten(a::ScalMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return sqrt(a.value) * x
end

### quadratic forms

function quad(a::ScalMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    if x isa AbstractVector
        return sum(abs2, x) * a.value
    else
        # map(Base.Fix1(quad, a), eachcol(x)) or similar alternatives
        # do NOT return a `SVector` for inputs `x::SMatrix`.
        wsq = let w = a.value
            x -> w * abs2(x)
        end 
        return vec(sum(wsq, x; dims=1))
    end
end

function quad!(r::AbstractArray, a::ScalMat, x::AbstractMatrix)
    @check_argdims eachindex(r) == axes(x, 2)
    @check_argdims a.dim == size(x, 1)
    @inbounds for i in axes(x, 2)
        r[i] = quad(a, view(x, :, i))
    end
    return r
end

function invquad(a::ScalMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    if x isa AbstractVector
        return sum(abs2, x) / a.value
    else
        # map(Base.Fix1(invquad, a), eachcol(x)) or similar alternatives
        # do NOT return a `SVector` for inputs `x::SMatrix`.
        wsq = let w = a.value
            x -> abs2(x) / w
        end 
        return vec(sum(wsq, x; dims=1))
    end
end

function invquad!(r::AbstractArray, a::ScalMat, x::AbstractMatrix)
    @check_argdims eachindex(r) == axes(x, 2)
    @check_argdims a.dim == size(x, 1)
    @inbounds for i in axes(x, 2)
        r[i] = invquad(a, view(x, :, i))
    end
    return r
end


### tri products

function X_A_Xt(a::ScalMat, x::AbstractMatrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 2)
    return Symmetric(a.value * (x * transpose(x)))
end

function Xt_A_X(a::ScalMat, x::AbstractMatrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 1)
    return Symmetric(a.value * (transpose(x) * x))
end

function X_invA_Xt(a::ScalMat, x::AbstractMatrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 2)
    return Symmetric((x * transpose(x)) / a.value)
end

function Xt_invA_X(a::ScalMat, x::AbstractMatrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 1)
    return Symmetric((transpose(x) * x) / a.value)
end

# Specializations for `x::Matrix` with reduced allocations
function X_A_Xt(a::ScalMat, x::Matrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 2)
    return Symmetric(lmul!(a.value, x * transpose(x)))
end

function Xt_A_X(a::ScalMat, x::Matrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 1)
    return Symmetric(lmul!(a.value, transpose(x) * x))
end

function X_invA_Xt(a::ScalMat, x::Matrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 2)
    return Symmetric(_rdiv!(x * transpose(x), a.value))
end

function Xt_invA_X(a::ScalMat, x::Matrix{<:Real})
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 1)
    return Symmetric(_rdiv!(transpose(x) * x, a.value))
end
