# positive diagonal matrix

immutable PDiagMat{T<:AbstractFloat,V<:AbstractVector} <: AbstractPDMat{T}
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


### Basics

dim(a::PDiagMat) = a.dim
full(a::PDiagMat) = diagm(a.diag)
diag(a::PDiagMat) = copy(a.diag)


### Arithmetics

function pdadd!{T<:AbstractFloat}(r::Matrix{T}, a::Matrix{T}, b::PDiagMat{T}, c::T)
    @check_argdims size(r) == size(a) == size(b)
    if is(r, a)
        _adddiag!(r, b.diag, c)
    else
        _adddiag!(copy!(r, a), b.diag, c)
    end
    return r
end

*{T<:AbstractFloat}(a::PDiagMat{T}, c::T) = PDiagMat(a.diag * c)
*{T<:AbstractFloat}(a::PDiagMat{T}, x::StridedVecOrMat{T}) = a.diag .* x
\{T<:AbstractFloat}(a::PDiagMat{T}, x::StridedVecOrMat{T}) = a.inv_diag .* x


### Algebra

inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
logdet(a::PDiagMat) = sum(log(a.diag))
eigmax(a::PDiagMat) = maximum(a.diag)
eigmin(a::PDiagMat) = minimum(a.diag)


### whiten and unwhiten

function whiten!{T<:AbstractFloat}(r::StridedVector{T}, a::PDiagMat{T}, x::StridedVector{T})
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.inv_diag
    for i = 1:n
        r[i] = x[i] * sqrt(v[i])
    end
    return r
end

function unwhiten!{T<:AbstractFloat}(r::StridedVector{T}, a::PDiagMat{T}, x::StridedVector{T})
    n = dim(a)
    @check_argdims length(r) == length(x) == n
    v = a.diag
    for i = 1:n
        r[i] = x[i] * sqrt(v[i])
    end
    return r
end

whiten!{T<:AbstractFloat}(r::StridedMatrix{T}, a::PDiagMat{T}, x::StridedMatrix{T}) =
    broadcast!(*, r, x, sqrt(a.inv_diag))

unwhiten!{T<:AbstractFloat}(r::StridedMatrix{T}, a::PDiagMat{T}, x::StridedMatrix{T}) =
    broadcast!(*, r, x, sqrt(a.diag))


### quadratic forms

quad{T<:AbstractFloat}(a::PDiagMat{T}, x::Vector{T}) = wsumsq(a.diag, x)
invquad{T<:AbstractFloat}(a::PDiagMat{T}, x::Vector{T}) = wsumsq(a.inv_diag, x)

quad!{T<:AbstractFloat}(r::AbstractArray{T}, a::PDiagMat{T}, x::Matrix{T}) = At_mul_B!(r, abs2(x), a.diag)
invquad!{T<:AbstractFloat}(r::AbstractArray{T}, a::PDiagMat{T}, x::Matrix{T}) = At_mul_B!(r, abs2(x), a.inv_diag)


### tri products

function X_A_Xt{T<:AbstractFloat}(a::PDiagMat{T}, x::StridedMatrix{T})
    z = x .* reshape(sqrt(a.diag), 1, dim(a))
    A_mul_Bt(z, z)
end

function Xt_A_X{T<:AbstractFloat}(a::PDiagMat{T}, x::StridedMatrix{T})
    z = x .* sqrt(a.diag)
    At_mul_B(z, z)
end

function X_invA_Xt{T<:AbstractFloat}(a::PDiagMat{T}, x::StridedMatrix{T})
    z = x .* reshape(sqrt(a.inv_diag), 1, dim(a))
    A_mul_Bt(z, z)
end

function Xt_invA_X{T<:AbstractFloat}(a::PDiagMat{T}, x::StridedMatrix{T})
    z = x .* sqrt(a.inv_diag)
    At_mul_B(z, z)
end
