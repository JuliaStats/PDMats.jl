"""
Positive definite diagonal matrix.
"""
struct PDiagMat{T<:Real,V<:AbstractVector{T}} <: AbstractPDMat{T}
    diag::V
end

function Base.getproperty(a::PDiagMat, s::Symbol)
    if s === :dim
        return length(getfield(a, :diag))
    end
    return getfield(a, s)
end
Base.propertynames(::PDiagMat) = (:diag, :dim)

AbstractPDMat(A::Diagonal{<:Real}) = PDiagMat(A.diag)
AbstractPDMat(A::Symmetric{<:Real,<:Diagonal{<:Real}}) = PDiagMat(A.data.diag)
AbstractPDMat(A::Hermitian{<:Real,<:Diagonal{<:Real}}) = PDiagMat(A.data.diag)

### Conversion
Base.convert(::Type{PDiagMat{T}}, a::PDiagMat{T}) where {T<:Real} = a
function Base.convert(::Type{PDiagMat{T}}, a::PDiagMat) where {T<:Real}
    diag = convert(AbstractVector{T}, a.diag)
    return PDiagMat{T,typeof(diag)}(diag)
end
Base.convert(::Type{AbstractPDMat{T}}, a::PDiagMat) where {T<:Real} = convert(PDiagMat{T}, a)

### Basics

Base.size(a::PDiagMat) = (a.dim, a.dim)
Base.Matrix{T}(a::PDiagMat) where {T} = Matrix{T}(Diagonal(a.diag))
LinearAlgebra.diag(a::PDiagMat) = copy(a.diag)
LinearAlgebra.cholesky(a::PDiagMat) = Cholesky(Diagonal(map(sqrt, a.diag)), 'U', 0)

### Treat as a `Diagonal` matrix in broadcasting since that is better supported
Base.broadcastable(a::PDiagMat) = Base.broadcastable(Diagonal(a.diag))

### Inheriting from AbstractMatrix

Base.@propagate_inbounds Base.getindex(a::PDiagMat{T}, i::Int, j::Int) where {T} = i == j ? a.diag[i] : zero(T)

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
    @check_argdims a.dim == length(x)
    return a.diag .* x
end
function *(a::PDiagMat, x::AbstractMatrix)
    @check_argdims a.dim == size(x, 1)
    return a.diag .* x
end
function \(a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return x ./ a.diag
end
function /(x::AbstractVecOrMat, a::PDiagMat)
    @check_argdims a.dim == size(x, 2)
    if VERSION < v"1.9-"
        # return matrix for 1-element vectors `x`, consistent with LinearAlgebra < 1.9
        return reshape(x, Val(2)) ./ permutedims(a.diag) # = (a' \ x')'
    else
        return x ./ (x isa AbstractVector ? a.diag : a.diag')
    end
end
Base.kron(A::PDiagMat, B::PDiagMat) = PDiagMat(vec(permutedims(A.diag) .* B.diag))

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

function whiten!(r::AbstractVecOrMat, a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    return r .= x ./ sqrt.(a.diag)
end
function unwhiten!(r::AbstractVecOrMat, a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims axes(r) == axes(x)
    @check_argdims a.dim == size(x, 1)
    return r .= x .* sqrt.(a.diag)
end

function whiten(a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return x ./ sqrt.(a.diag)
end
function unwhiten(a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    return x .* sqrt.(a.diag)
end

### quadratic forms

function quad(a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    if x isa AbstractVector
        return wsumsq(a.diag, x)
    else
        # map(Base.Fix1(invquad, a), eachcol(x)) or similar alternatives
        # do NOT return a `SVector` for inputs `x::SMatrix`.
        return vec(sum(abs2.(x) .* a.diag; dims = 1))
    end
end

function quad!(r::AbstractArray, a::PDiagMat, x::AbstractMatrix)
    ad = a.diag
    @check_argdims eachindex(ad) == axes(x, 1)
    @check_argdims eachindex(r) == axes(x, 2)
    @inbounds for j in axes(x, 2)
        s = zero(promote_type(eltype(ad), eltype(x)))
        for i in axes(x, 1)
            s += ad[i] * abs2(x[i,j])
        end
        r[j] = s
    end
    r
end

function invquad(a::PDiagMat, x::AbstractVecOrMat)
    @check_argdims a.dim == size(x, 1)
    if x isa AbstractVector
        return invwsumsq(a.diag, x)
    else
        # map(Base.Fix1(invquad, a), eachcol(x)) or similar alternatives
        # do NOT return a `SVector` for inputs `x::SMatrix`.
        return vec(sum(abs2.(x) ./ a.diag; dims = 1))
    end
end

function invquad!(r::AbstractArray, a::PDiagMat, x::AbstractMatrix)
    ad = a.diag
    @check_argdims eachindex(ad) == axes(x, 1)
    @check_argdims eachindex(r) == axes(x, 2)
    @inbounds for j in axes(x, 2)
        s = zero(zero(eltype(x)) / zero(eltype(ad)))
        for i in axes(x, 1)
            s += abs2(x[i,j]) / ad[i]
        end
        r[j] = s
    end
    r
end


### tri products

function X_A_Xt(a::PDiagMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = a.diag .* transpose(x)
    return Symmetric(x * z)
end

function Xt_A_X(a::PDiagMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = a.diag .* x
    return Symmetric(transpose(x) * z)
end

function X_invA_Xt(a::PDiagMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 2)
    z = transpose(x) ./ a.diag
    return Symmetric(x * z)
end

function Xt_invA_X(a::PDiagMat, x::AbstractMatrix{<:Real})
    @check_argdims a.dim == size(x, 1)
    z = x ./ a.diag
    return Symmetric(transpose(x) * z)
end

### Specializations for `Array` arguments with reduced allocations

function quad(a::PDiagMat{<:Real,<:Vector}, x::Matrix)
    @check_argdims a.dim == size(x, 1)
    T = typeof(zero(eltype(a)) * abs2(zero(eltype(x))))
    return quad!(Vector{T}(undef, size(x, 2)), a, x)
end

function invquad(a::PDiagMat{<:Real,<:Vector}, x::Matrix)
    @check_argdims a.dim == size(x, 1)
    T = typeof(abs2(zero(eltype(x))) / zero(eltype(a)))
    return invquad!(Vector{T}(undef, size(x, 2)), a, x)
end

