module PDMats

using ArrayViews

# using NumericExtensions

import Base: +, *, \, /, ==
import Base: full, logdet, inv, diag, diagm

export
    AbstractPDMat, PDMat, PDiagMat, ScalMat, 
    dim, full, whiten, whiten!, unwhiten, unwhiten!, add_scal!, add_scal,
    quad, quad!, invquad, invquad!, X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X,
    unwhiten_winv!, unwhiten_winv

import Base.BLAS: nrm2, axpy!, gemv!, gemm, gemm!, trmv, trmv!, trmm, trmm!
import Base.LAPACK: trtrs!
import Base.LinAlg: A_ldiv_B!, A_mul_B!, A_mul_Bc!, A_rdiv_B!, A_rdiv_Bc!, Ac_ldiv_B!, Cholesky

macro check_argdims(cond)
    quote
        ($(cond)) || error("Inconsistent argument dimensions.")
    end
end

#################################################
#
#   auxiliary functions
#
#################################################

function sumsq{T}(a::AbstractArray{T})
    s = zero(T)
    for i = 1:length(a)
        @inbounds s += abs2(a[i])
    end
    return s
end

function wsumsq(w::AbstractVector, a::AbstractVector)
    @check_argdims(length(a) == length(w))
    s = 0.
    for i = 1:length(a)
        @inbounds s += abs2(a[i]) * w[i]
    end
    return s
end

function mulcols!{T}(r::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractVector{T}) 
    # multiple b to each column of a
    m = size(a, 1)
    n = size(a, 2)
    @check_argdims(size(r) == (m, n) && length(b) == m)
    for j = 1:n
        aj = view(a, :, j)
        rj = view(r, :, j)
        for i = 1:m
            @inbounds rj[i] = aj[i] * b[i]
        end
    end
    r
end

mulcols!{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulcols!(a, a, b)
mulcols{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulcols!(similar(a), a, b)

function mulrows!{T}(r::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractVector{T}) 
    # multiple b to each column of a
    m = size(a, 1)
    n = size(a, 2)
    @check_argdims(size(r) == (m, n) && length(b) == n)
    for j = 1:n
        aj = view(a, :, j)
        rj = view(r, :, j)
        bj = b[j]
        for i = 1:m
            @inbounds rj[i] = aj[i] * bj
        end
    end
    r
end

mulrows!{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulrows!(a, a, b)
mulrows{T}(a::AbstractMatrix{T}, b::AbstractVector{T}) = mulrows!(similar(a), a, b)

function mulsqrt(x::Vector, c::Vector) 
    @check_argdims length(x) == length(c)
    [x[i] * sqrt(c[i]) for i in 1 : length(x)]
end

function mulsqrt!(x::Vector, c::Vector)
    @check_argdims length(x) == length(c)
    for i in 1 : length(x)
        x[i] .*= sqrt(c[i])
    end
    x
end

function add_diag!(a::Matrix, v::Number)
    n = minimum(size(a))::Int
    for i = 1:n
        @inbounds a[i,i] += v
    end
    a
end

function add_diag!(a::Matrix, v::Vector)
    n = minimum(size(a))::Int
    @check_argdims length(v) == n
    for i = 1:n
        @inbounds a[i,i] += v[i]
    end
    a
end

function add_diag!(a::Matrix, v::Vector, c::Number)
    n = minimum(size(a))::Int
    @check_argdims length(v) == n
    for i = 1:n
        @inbounds a[i,i] += v[i] * c
    end
    a
end

add_diag(a::Matrix, v::Number) = add_diag!(copy(a), v)
add_diag(a::Matrix, v::Vector) = add_diag!(copy(a), v)
add_diag(a::Matrix, v::Vector, c::Number) = add_diag!(copy(a), v, c)

#################################################
#
#   PDMat: full pos. def. matrix
#
#################################################

abstract AbstractPDMat

immutable PDMat <: AbstractPDMat
    dim::Int
    mat::Matrix{Float64}    
    chol::Cholesky{Float64}    
    
end

function PDMat(mat::Matrix{Float64})
    d = size(mat, 1)
    if !(d >= 1 && size(mat, 2) == d)
        throw(ArgumentError("mat must be a square matrix."))
    end
    PDMat(d, mat, cholfact(mat))
end
PDMat(fac::Cholesky) = PDMat(size(fac,1), full(fac), fac)
PDMat(mat::Symmetric{Float64}) = PDMat(full(mat))

# basics

Base.size(a::PDMat) = (a.dim,a.dim)
Base.size(a::PDMat,i) = size(a.mat,i)
dim(a::PDMat) = a.dim
full(a::PDMat) = copy(a.mat)
inv(a::PDMat) = PDMat(inv(a.chol))
logdet(a::PDMat) = logdet(a.chol)
diag(a::PDMat) = diag(a.mat)

* (a::PDMat, c::Float64) = PDMat(a.mat * c)
* (a::PDMat, x::StridedVecOrMat) = a.mat * x
\ (a::PDMat, x::StridedVecOrMat) = a.chol \ x

# whiten and unwhiten

## for a.chol.uplo == 'U', a.chol[:U] does not copy.
## Similarly a.chol[:L] does not copy when a.chol.uplo == 'L'
function whiten!(a::PDMat, x::StridedVecOrMat{Float64})
    a.chol.uplo == 'U' ? Ac_ldiv_B!(a.chol[:U], x) : A_ldiv_B!(a.chol[:L], x)
end
whiten(a::PDMat, x::StridedVecOrMat{Float64}) = whiten!(a, copy(x))

function unwhiten!(a::PDMat, x::StridedVecOrMat{Float64})
    a.chol.uplo == 'U' ? Ac_mul_B!(a.chol[:U],x) : A_mul_B!(a.chol[:L], x)
end
unwhiten(a::PDMat, x::StridedVecOrMat{Float64}) = unwhiten!(a, copy(x))

function unwhiten_winv!(a::PDMat, x::StridedVecOrMat{Float64})
    a.chol.uplo == 'U' ? A_mul_B!(a.chol[:U], x) : Ac_mul_B!(a.chol[:L], x)
end
unwhiten_winv(a::PDMat, x::StridedVecOrMat{Float64}) = unwhiten_winv!(a, copy(x))

# quadratic forms

quad(a::PDMat, x::Vector{Float64}) = dot(x, a.mat * x)
invquad(a::PDMat, x::Vector{Float64}) = abs2(norm(whiten(a, x)))
    
function quad!(r::Array{Float64}, a::PDMat, x::Matrix{Float64}) # = dot!(r, x, a.mat * x, 1)
    n = size(x,2)
    @check_argdims(length(r) == n)
    ax = a.mat * x
    for j = 1:n
        r[j] = dot(view(ax, :, j), view(x, :, j))
    end
    r
end

function invquad!(r::Array{Float64}, a::PDMat, x::Matrix{Float64}) # = sumsq!(fill!(r, 0.0), whiten(a, x), 1)
    n = size(x, 2)
    @check_argdims(length(r) == n)
    wx = whiten(a, x)
    for j = 1:n
        r[j] = sumsq(view(wx, :, j))
    end
    return r
end

function X_A_Xt(a::PDMat, x::Matrix{Float64})
    z = copy(x) # dimension checks will be done in the A_mul_B*! methods
    a.chol.uplo == 'U' ? A_mul_Bc!(z, a.chol[:U]) : A_mul_B!(z, a.chol[:L])
    gemm('N', 'T', 1.0, z, z)
end

function Xt_A_X(a::PDMat, x::Matrix{Float64})
    z = copy(x)
    a.chol.uplo == 'U' ? A_mul_B!(a.chol[:U], z) : Ac_mul_B!(a.chol[:L], z)
    gemm('T', 'N', 1.0, z, z)
end

function X_invA_Xt(a::PDMat, x::Matrix{Float64})
    z = copy(x)
    a.chol.uplo == 'U' ? A_rdiv_B!(z, a.chol[:U]) : A_rdiv_Bc!(z, a.chol[:L])
    gemm('N','T', 1.0, z, z)
end

function Xt_invA_X(a::PDMat, x::Matrix{Float64})
    z = copy(x)
    a.chol.uplo == 'U' ? Ac_ldiv_B!(a.chol[:U], z) : A_ldiv_B!(a.chol[:L], z)
    gemm('T', 'N', 1.0, z, z)
end


#################################################
#
#   PDiagMat: positive diagonal matrix
#
#################################################

immutable PDiagMat <: AbstractPDMat
    dim::Int
    diag::Vector{Float64}
    inv_diag::Vector{Float64}
    
    PDiagMat(v::Vector{Float64}) = new(length(v), v, 1.0 ./ v)    
    
    function PDiagMat(v::Vector{Float64}, inv_v::Vector{Float64})
        @check_argdims length(v) == length(inv_v)
        new(length(v), v, inv_v)
    end
end

# basics

Base.size(a::PDiagMat) = (a.dim,a.dim)
Base.size(a::PDiagMat,i) = (i < 1) ? error("dimension out of range") : (i < 3 ? a.dim : 1)
dim(a::PDiagMat) = a.dim
full(a::PDiagMat) = diagm(a.diag)
inv(a::PDiagMat) = PDiagMat(a.inv_diag, a.diag)
logdet(a::PDiagMat) = sum(log(a.diag))
diag(a::PDiagMat) = copy(a.diag)

* (a::PDiagMat, c::Float64) = PDiagMat(a.diag * c)
* (a::PDiagMat, x::Vector{Float64}) = a.diag .* x
\ (a::PDiagMat, x::Vector{Float64}) = a.inv_diag .* x
* (a::PDiagMat, x::Matrix{Float64}) = mulcols(x, a.diag)
\ (a::PDiagMat, x::Matrix{Float64}) = mulcols(x, a.inv_diag)

# whiten and unwhiten 

whiten(a::PDiagMat, x::Vector{Float64}) = mulsqrt(x, a.inv_diag)
whiten(a::PDiagMat, x::Matrix{Float64}) = mulcols(x, sqrt(a.inv_diag))

whiten!(a::PDiagMat, x::Vector{Float64}) = mulsqrt!(x, a.inv_diag)
whiten!(a::PDiagMat, x::Matrix{Float64}) = mulcols!(x, sqrt(a.inv_diag))

unwhiten(a::PDiagMat, x::Vector{Float64}) = mulsqrt(x, a.diag)
unwhiten(a::PDiagMat, x::Matrix{Float64}) = mulcols(x, sqrt(a.diag))

unwhiten!(a::PDiagMat, x::Vector{Float64}) = mulsqrt!(x, a.diag)
unwhiten!(a::PDiagMat, x::Matrix{Float64}) = mulcols!(x, sqrt(a.diag))

unwhiten_winv!(J::PDiagMat, z::StridedVecOrMat{Float64}) = whiten!(J, z)
unwhiten_winv(J::PDiagMat, z::StridedVecOrMat{Float64}) = whiten(J, z)

# quadratic forms

quad(a::PDiagMat, x::Vector{Float64}) = wsumsq(a.diag, x)
invquad(a::PDiagMat, x::Vector{Float64}) = wsumsq(a.inv_diag, x)

quad!(r::Array{Float64}, a::PDiagMat, x::Matrix{Float64}) = gemv!('T', 1., x .* x, a.diag, 0., r)
invquad!(r::Array{Float64}, a::PDiagMat, x::Matrix{Float64}) = gemv!('T', 1., x .* x, a.inv_diag, 0., r)

function X_A_Xt(a::PDiagMat, x::Matrix{Float64}) 
    z = mulrows(x, sqrt(a.diag))
    gemm('N', 'T', 1.0, z, z)
end

function Xt_A_X(a::PDiagMat, x::Matrix{Float64})
    z = mulcols(x, sqrt(a.diag))
    gemm('T', 'N', 1.0, z, z)
end

function X_invA_Xt(a::PDiagMat, x::Matrix{Float64})
    z = mulrows(x, sqrt(a.inv_diag))
    gemm('N', 'T', 1.0, z, z)
end

function Xt_invA_X(a::PDiagMat, x::Matrix{Float64})
    z = mulcols(x, sqrt(a.inv_diag))
    gemm('T', 'N', 1.0, z, z)
end


#################################################
#
#   ScalMat: s * eye(d) with s > 0
#
#################################################

immutable ScalMat <: AbstractPDMat
    dim::Int
    value::Float64
    inv_value::Float64
    
    ScalMat(d::Int, v::Float64) = new(d, v, 1.0 / v)
    ScalMat(d::Int, v::Float64, inv_v::Float64) = new(d, v, inv_v)
end

# basics

dim(a::ScalMat) = a.dim
full(a::ScalMat) = diagm(fill(a.value, a.dim))
inv(a::ScalMat) = ScalMat(a.dim, a.inv_value, a.value)
logdet(a::ScalMat) = a.dim * log(a.value)
diag(a::ScalMat) = fill(a.value, a.dim)

* (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value * c)
/ (a::ScalMat, c::Float64) = ScalMat(a.dim, a.value / c)
* (a::ScalMat, x::StridedVecOrMat) = a.value * x
\ (a::ScalMat, x::StridedVecOrMat) = a.inv_value * x

# whiten and unwhiten 

function whiten(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    x * sqrt(a.inv_value)
end

function whiten!(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    sv = sqrt(a.inv_value)
    for i = 1:length(x)
        @inbounds x[i] *= sv
    end
    x
end

function unwhiten(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    x * sqrt(a.value)
end

function unwhiten!(a::ScalMat, x::StridedVecOrMat{Float64})
    @check_argdims dim(a) == size(x, 1)
    sv = sqrt(a.value)
    for i = 1:length(x)
        @inbounds x[i] *= sv
    end
    x
end

unwhiten_winv!(J::ScalMat,  z::StridedVecOrMat{Float64}) = whiten!(J, z)
unwhiten_winv(J::ScalMat, z::StridedVecOrMat{Float64}) = whiten(J, z)

# quadratic forms

function quad(a::ScalMat, x::Vector{Float64})
    @check_argdims dim(a) == size(x, 1)
    abs2(nrm2(x)) * a.value
end

function invquad(a::ScalMat, x::Vector{Float64})
    @check_argdims dim(a) == size(x, 1)
    abs2(nrm2(x)) * a.inv_value
end

function quad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims dim(a) == m && length(r) == n
    for j = 1:n
        r[j] = sumsq(view(x, :, j)) * a.value
    end
    r
end

function invquad!(r::AbstractArray{Float64}, a::ScalMat, x::Matrix{Float64})
    m = size(x, 1)
    n = size(x, 2)
    @check_argdims dim(a) == m && length(r) == n
    for j = 1:n
        r[j] = sumsq(view(x, :, j)) * a.inv_value
    end
    r
end

function X_A_Xt(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.value, x, x)
end

function Xt_A_X(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.value, x, x)
end

function X_invA_Xt(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 2)
    gemm('N', 'T', a.inv_value, x, x)
end

function Xt_invA_X(a::ScalMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    gemm('T', 'N', a.inv_value, x, x)
end


#################################################
#
#   generic functions for p.d. matrices
#
#################################################

* (c::Float64, a::AbstractPDMat) = a * c
/ (a::AbstractPDMat, c::Float64) = a * inv(c)

function quad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    r = Array(Float64, size(x,2))
    quad!(r, a, x)
    r
end

function invquad(a::AbstractPDMat, x::Matrix{Float64})
    @check_argdims dim(a) == size(x, 1)
    r = Array(Float64, size(x,2))
    invquad!(r, a, x)
    r
end


#################################################
#
#   addition
#
#################################################

# addition between p.d. matrices and ordinary ones

+ (a::PDMat,    b::Matrix{Float64}) = a.mat + b
+ (a::PDiagMat, b::Matrix{Float64}) = add_diag(b, a.diag)
+ (a::ScalMat,  b::Matrix{Float64}) = add_diag(b, a.value)

+ (a::Matrix{Float64}, b::AbstractPDMat) = b + a

function add!(a::Matrix{Float64}, b::PDMat)
    bm = b.mat
    @check_argdims size(a) == size(bm)
    for i = 1:length(a)
        @inbounds a[i] += bm[i]
    end
    a
end

add!(a::Matrix{Float64}, b::PDiagMat) = add_diag!(a, b.diag)
add!(a::Matrix{Float64}, b::ScalMat) = add_diag!(a, b.value)

add_scal!(a::Matrix{Float64}, b::PDMat, c::Float64) = axpy!(c, b.mat, a)
add_scal!(a::Matrix{Float64}, b::PDiagMat, c::Float64) = add_diag!(a, b.diag, c)
add_scal!(a::Matrix{Float64}, b::ScalMat, c::Float64) = add_diag!(a, b.value * c)

add_scal(a::Matrix{Float64}, b::AbstractPDMat, c::Float64) = add_scal!(copy(a), b, c)

# between pdmat and pdmat

+ (a::PDMat, b::AbstractPDMat) = PDMat(a.mat + full(b))
+ (a::PDiagMat, b::AbstractPDMat) = PDMat(add_diag!(full(b), a.diag))
+ (a::ScalMat, b::AbstractPDMat) = PDMat(add_diag!(full(b), a.value))

+ (a::PDMat, b::PDMat) = PDMat(a.mat + b.mat)
+ (a::PDMat, b::PDiagMat) = PDMat(add_diag(a.mat, b.diag))
+ (a::PDMat, b::ScalMat) = PDMat(add_diag(a.mat, b.value))

+ (a::PDiagMat, b::PDMat) = PDMat(add_diag(b.mat, a.diag))
+ (a::PDiagMat, b::PDiagMat) = PDiagMat(a.diag + b.diag)
+ (a::PDiagMat, b::ScalMat) = PDiagMat(a.diag .+ b.value)

+ (a::ScalMat, b::PDMat) = PDMat(add_diag(b.mat, a.value))
+ (a::ScalMat, b::PDiagMat) = PDiagMat(a.value .+ b.diag)
+ (a::ScalMat, b::ScalMat) = ScalMat(a.dim, a.value + b.value)

add_scal(a::PDMat, b::AbstractPDMat, c::Float64) = PDMat(a.mat + full(b * c))
add_scal(a::PDiagMat, b::AbstractPDMat, c::Float64) = PDMat(add_diag!(full(b * c), a.diag))
add_scal(a::ScalMat, b::AbstractPDMat, c::Float64) = PDMat(add_diag!(full(b * c), a.value))

add_scal(a::PDMat, b::PDMat, c::Float64) = PDMat(a.mat + b.mat * c)
add_scal(a::PDMat, b::PDiagMat, c::Float64) = PDMat(add_diag(a.mat, b.diag, c))
add_scal(a::PDMat, b::ScalMat, c::Float64) = PDMat(add_diag(a.mat, b.value * c))

add_scal(a::PDiagMat, b::PDMat, c::Float64) = PDMat(add_diag!(b.mat * c, a.diag))
add_scal(a::PDiagMat, b::PDiagMat, c::Float64) = PDiagMat(a.diag + b.diag * c)
add_scal(a::PDiagMat, b::ScalMat, c::Float64) = PDiagMat(a.diag .+ b.value * c)

add_scal(a::ScalMat, b::PDMat, c::Float64) = PDMat(add_diag!(b.mat * c, a.value))
add_scal(a::ScalMat, b::PDiagMat, c::Float64) = PDiagMat(a.value .+ b.diag * c)
add_scal(a::ScalMat, b::ScalMat, c::Float64) = ScalMat(a.dim, a.value + b.value * c)

end # module
