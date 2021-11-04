"""
Positive definite diagonal matrix.
"""
struct PDiagMat{T<:Real,V<:AbstractVector} <: AbstractPDMat{T}
    dim::Int
    diag::V

    PDiagMat{T,S}(d::Int,v::AbstractVector) where {T,S} =
        new{T,S}(d,v)
    
    # deprecated
    PDiagMat{T,S}(d::Int,v::AbstractVector,inv_v::AbstractVector) where {T,S} =
        new{T,S}(d,v)
end

function Base.getproperty(pdm::PDiagMat, s::Symbol)
    if s == :inv_diag
        # deprecated
        return inv.(Base.getfield(pdm,:diag))
    else
        return Base.getfield(pdm,s)
    end
end

function PDiagMat(v::AbstractVector)
    @check_argdims length(v) == length(inv.(v))
    PDiagMat{eltype(v),typeof(v)}(length(v), v)
end

# deprecated
function PDiagMat(v::AbstractVector,inv_v::AbstractVector)
    @check_argdims length(v) == length(inv_v)
    PDiagMat{eltype(v),typeof(v)}(length(v), v, inv_v)
end

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

*(a::PDiagMat, c::T) where {T<:Real} = PDiagMat(a.diag * c)
*(a::PDiagMat, x::AbstractVector) = a.diag .* x
*(a::PDiagMat, x::AbstractMatrix) = a.diag .* x
\(a::PDiagMat, x::AbstractVecOrMat) = a.diag .\ x
/(x::AbstractVecOrMat, a::PDiagMat) = a.diag ./ x
Base.kron(A::PDiagMat, B::PDiagMat) = PDiagMat( vcat([A.diag[i] * B.diag for i in 1:dim(A)]...) )

### Algebra

Base.inv(a::PDiagMat) = PDiagMat(inv.(a.diag))
function LinearAlgebra.logdet(a::PDiagMat)
    diag = a.diag
    return isempty(diag) ? zero(log(zero(eltype(diag)))) : sum(log, diag)
end
LinearAlgebra.eigmax(a::PDiagMat) = maximum(a.diag)
LinearAlgebra.eigmin(a::PDiagMat) = minimum(a.diag)


### whiten and unwhiten

function whiten!(r::StridedVector, a::PDiagMat, x::StridedVector)
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = inv.(a.diag)
    for i = 1:n
        r[i] = x[i] * sqrt(v[i])
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

whiten!(r::StridedMatrix, a::PDiagMat, x::StridedMatrix) =
    broadcast!(*, r, x, sqrt.(inv.(a.diag)))

unwhiten!(r::StridedMatrix, a::PDiagMat, x::StridedMatrix) =
    broadcast!(*, r, x, sqrt.(a.diag))


### quadratic forms

quad(a::PDiagMat, x::AbstractVector) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::AbstractVector) = wsumsq(inv.(a.diag), x)

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
    ainvd = inv.(a.diag)
    @check_argdims m == length(ainvd) && length(r) == n
    @inbounds for j = 1:n
        s = zero(promote_type(eltype(ainvd), eltype(x)))
        for i in 1:m
            s += ainvd[i] * abs2(x[i,j])
        end
        r[j] = s
    end
    r
end


### tri products

function X_A_Xt(a::PDiagMat, x::StridedMatrix)
    z = x .* reshape(sqrt.(a.diag), 1, dim(a))
    z * transpose(z)
end

function Xt_A_X(a::PDiagMat, x::StridedMatrix)
    z = x .* sqrt.(a.diag)
    transpose(z) * z
end

function X_invA_Xt(a::PDiagMat, x::StridedMatrix)
    z = x .* reshape(sqrt.(inv.(a.diag)), 1, dim(a))
    z * transpose(z)
end

function Xt_invA_X(a::PDiagMat, x::StridedMatrix)
    z = x .* sqrt.(inv.(a.diag))
    transpose(z) * z
end
