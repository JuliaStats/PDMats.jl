# positive diagonal matrix

immutable PDiagMat{T<:Real,V<:AbstractVector} <: AbstractPDMat{T}
  dim::Int
  diag::V
  inv_diag::V
  PDiagMat(d::Int,v::AbstractVector{T},inv_v::AbstractVector{T}) = new(d,v,inv_v)
end

function PDiagMat(v::AbstractVector,inv_v::AbstractVector)
  @check_argdims length(v) == length(inv_v)
  PDiagMat{eltype(v),typeof(v)}(length(v), v, inv_v)
end

PDiagMat(v::Vector) = PDiagMat(v, ones(v)./v)

### Conversion
convert{T<:Real}(::Type{PDiagMat{T}},      a::PDiagMat) = PDiagMat(convert(AbstractArray{T}, a.diag))
convert{T<:Real}(::Type{AbstractArray{T}}, a::PDiagMat) = convert(PDiagMat{T}, a)

### Basics

dim(a::PDiagMat) = a.dim
full(a::PDiagMat) = diagm(a.diag)
diag(a::PDiagMat) = copy(a.diag)


### Arithmetics

function pdadd!(r::Matrix, a::Matrix, b::PDiagMat, c)
    @check_argdims size(r) == size(a) == size(b)
    if r === a
        _adddiag!(r, b.diag, c)
    else
        _adddiag!(copy!(r, a), b.diag, c)
    end
    return r
end

*{T<:Real}(a::PDiagMat, c::T) = PDiagMat(a.diag * c)
*(a::PDiagMat, x::StridedVecOrMat) = a.diag .* x
\(a::PDiagMat, x::StridedVecOrMat) = a.inv_diag .* x


### Algebra

inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
logdet(a::PDiagMat) = sum(log, a.diag)
eigmax(a::PDiagMat) = maximum(a.diag)
eigmin(a::PDiagMat) = minimum(a.diag)


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
    broadcast!(*, r, x, @compat(sqrt.(a.inv_diag)))

unwhiten!(r::StridedMatrix, a::PDiagMat, x::StridedMatrix) =
    broadcast!(*, r, x, @compat(sqrt.(a.diag)))


### quadratic forms

quad(a::PDiagMat, x::Vector) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::Vector) = wsumsq(a.inv_diag, x)

quad!(r::AbstractArray, a::PDiagMat, x::Matrix) = At_mul_B!(r, @compat(abs2.(x)), a.diag)
invquad!(r::AbstractArray, a::PDiagMat, x::Matrix) = At_mul_B!(r, @compat(abs2.(x)), a.inv_diag)


### tri products

function X_A_Xt(a::PDiagMat, x::StridedMatrix)
    z = x .* reshape(@compat(sqrt.(a.diag)), 1, dim(a))
    A_mul_Bt(z, z)
end

function Xt_A_X(a::PDiagMat, x::StridedMatrix)
    z = x .* @compat(sqrt.(a.diag))
    At_mul_B(z, z)
end

function X_invA_Xt(a::PDiagMat, x::StridedMatrix)
    z = x .* reshape(@compat(sqrt.(a.inv_diag)), 1, dim(a))
    A_mul_Bt(z, z)
end

function Xt_invA_X(a::PDiagMat, x::StridedMatrix)
    z = x .* @compat(sqrt.(a.inv_diag))
    At_mul_B(z, z)
end
