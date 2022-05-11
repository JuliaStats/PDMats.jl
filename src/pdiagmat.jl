"""
Positive definite diagonal matrix.
"""
struct PDiagMat{T<:Real,V<:AbstractVector{T}} <: AbstractPDMat{T}
    dim::Int
    diag::V
end

PDiagMat(v::AbstractVector{<:Real}) = PDiagMat{eltype(v),typeof(v)}(length(v), v)

### Conversion
Base.convert(::Type{PDiagMat{T}},      a::PDiagMat) where {T<:Real} = PDiagMat(convert(AbstractArray{T}, a.diag))
Base.convert(::Type{AbstractArray{T}}, a::PDiagMat) where {T<:Real} = convert(PDiagMat{T}, a)

### Basics

dim(a::PDiagMat) = a.dim
Base.Matrix(a::PDiagMat) = Matrix(Diagonal(a.diag))
LinearAlgebra.diag(a::PDiagMat) = copy(a.diag)
LinearAlgebra.cholesky(a::PDiagMat) = cholesky(Diagonal(a.diag))

### Inheriting from AbstractMatrix

function Base.getindex(a::PDiagMat, i::Integer)
    ncol, nrow = fldmod1(i, a.dim)
    ncol == nrow ? a.diag[nrow] : zero(eltype(a))
end
Base.getindex(a::PDiagMat{T},i::Integer,j::Integer) where {T} = i == j ? a.diag[i] : zero(T)

### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::PDiagMat, c)
    @check_argdims size(r) == size(a) == size(b)
    if r === a
        _adddiag!(r, b.diag, c)
    else
        _adddiag!(copyto!(r, a), b.diag, c)
    end
    return r
end

*(a::PDiagMat, c::Real) = PDiagMat(a.diag * c)
function *(a::PDiagMat, x::AbstractVector)
    @check_argdims dim(a) == length(x)
    return a.diag .* x
end
function *(a::PDiagMat, x::AbstractMatrix)
    @check_argdims dim(a) == size(x, 1)
    return a.diag .* x
end
function \(a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims dim(a) == size(x, 1)
    return x ./ a.diag
end
function /(x::AbstractVecOrMat, a::PDiagMat)
    @check_argdims dim(a) == size(x, 2)
    # return matrix for 1-element vectors `x`, consistent with LinearAlgebra
    return reshape(x, Val(2)) ./ permutedims(a.diag) # = (a' \ x')'
end
Base.kron(A::PDiagMat, B::PDiagMat) = PDiagMat( vcat([A.diag[i] * B.diag for i in 1:dim(A)]...) )

### Algebra

Base.inv(a::PDiagMat) = PDiagMat(map(inv, a.diag))
LinearAlgebra.det(a::PDiagMat) = prod(a.diag)
function LinearAlgebra.logdet(a::PDiagMat)
    diag = a.diag
    return isempty(diag) ? zero(log(zero(eltype(diag)))) : sum(log, diag)
end
LinearAlgebra.eigmax(a::PDiagMat) = maximum(a.diag)
LinearAlgebra.eigmin(a::PDiagMat) = minimum(a.diag)
LinearAlgebra.sqrt(a::PDiagMat) = PDiagMat(map(sqrt, a.diag))


### whiten and unwhiten

function whiten!(r::StridedVector, a::PDiagMat, x::StridedVector)
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.diag
    for i = 1:n
        r[i] = x[i] / sqrt(v[i])
    end
    return r
end

function unwhiten!(r::StridedVector, a::PDiagMat, x::StridedVector)
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.diag
    for i = 1:n
        r[i] = x[i] * sqrt(v[i])
    end
    return r
end

function whiten!(r::StridedMatrix, a::PDiagMat, x::StridedMatrix)
    r .= x ./ sqrt.(a.diag)
    return r
end

function unwhiten!(r::StridedMatrix, a::PDiagMat, x::StridedMatrix)
    r .= x .* sqrt.(a.diag)
    return r
end


### quadratic forms

quad(a::PDiagMat, x::AbstractVector) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::AbstractVector) = invwsumsq(a.diag, x)

function quad!(r::AbstractArray, a::PDiagMat, x::StridedMatrix)
    m, n = size(x)
    ad = a.diag
    @check_argdims m == length(ad) && length(r) == n
    @inbounds for j = 1:n
        s = zero(promote_type(eltype(ad), eltype(x)))
        for i in 1:m
            s += ad[i] * abs2(x[i,j])
        end
        r[j] = s
    end
    r
end

function invquad!(r::AbstractArray, a::PDiagMat, x::StridedMatrix)
    m, n = size(x)
    ad = a.diag
    @check_argdims m == length(ad) && length(r) == n
    @inbounds for j = 1:n
        s = zero(zero(eltype(x)) / zero(eltype(ad)))
        for i in 1:m
            s += abs2(x[i,j]) / ad[i]
        end
        r[j] = s
    end
    r
end


### tri products

function X_A_Xt(a::PDiagMat, x::AbstractMatrix)
    @check_argdims dim(a) == size(x, 2)
    z = x .* sqrt.(permutedims(a.diag))
    z * transpose(z)
end

function Xt_A_X(a::PDiagMat, x::AbstractMatrix)
    @check_argdims dim(a) == size(x, 1)
    z = x .* sqrt.(a.diag)
    transpose(z) * z
end

function X_invA_Xt(a::PDiagMat, x::AbstractMatrix)
    @check_argdims dim(a) == size(x, 2)
    z = x ./ sqrt.(permutedims(a.diag))
    z * transpose(z)
end

function Xt_invA_X(a::PDiagMat, x::AbstractMatrix)
    @check_argdims dim(a) == size(x, 1)
    z = x ./ sqrt.(a.diag)
    transpose(z) * z
end
