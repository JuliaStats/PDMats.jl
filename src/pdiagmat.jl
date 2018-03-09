# positive diagonal matrix

struct PDiagMat{T<:Real,V<:AbstractVector} <: AbstractPDMat{T}
  dim::Int
  diag::V
  inv_diag::V
  PDiagMat{T,S}(d::Int,v::AbstractVector{T},inv_v::AbstractVector{T}) where {T,S} = new{T,S}(d,v,inv_v)
end

function PDiagMat(v::AbstractVector,inv_v::AbstractVector)
  @check_argdims length(v) == length(inv_v)
  PDiagMat{eltype(v),typeof(v)}(length(v), v, inv_v)
end

PDiagMat(v::Vector) = PDiagMat(v, inv.(v))

### Conversion
Base.convert(::Type{PDiagMat{T}},      a::PDiagMat) where {T<:Real} = PDiagMat(convert(AbstractArray{T}, a.diag))
Base.convert(::Type{AbstractArray{T}}, a::PDiagMat) where {T<:Real} = convert(PDiagMat{T}, a)

### Basics

dim(a::PDiagMat) = a.dim
Base.Matrix(a::PDiagMat) = Matrix(Diagonal(a.diag))
LinearAlgebra.diag(a::PDiagMat) = copy(a.diag)


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
*(a::PDiagMat, x::StridedVecOrMat) = a.diag .* x
\(a::PDiagMat, x::StridedVecOrMat) = a.inv_diag .* x


### Algebra

Base.inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
LinearAlgebra.logdet(a::PDiagMat) = sum(log, a.diag)
LinearAlgebra.eigmax(a::PDiagMat) = maximum(a.diag)
LinearAlgebra.eigmin(a::PDiagMat) = minimum(a.diag)


### whiten and unwhiten

function whiten!(r::StridedVector, a::PDiagMat, x::StridedVector)
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.inv_diag
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
    broadcast!(*, r, x, sqrt.(a.inv_diag))

unwhiten!(r::StridedMatrix, a::PDiagMat, x::StridedMatrix) =
    broadcast!(*, r, x, sqrt.(a.diag))


### quadratic forms

quad(a::PDiagMat, x::StridedVector) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::StridedVector) = wsumsq(a.inv_diag, x)

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
    ainvd = a.inv_diag
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
    z = x .* reshape(sqrt.(a.inv_diag), 1, dim(a))
    z * transpose(z)
end

function Xt_invA_X(a::PDiagMat, x::StridedMatrix)
    z = x .* sqrt.(a.inv_diag)
    transpose(z) * z
end
