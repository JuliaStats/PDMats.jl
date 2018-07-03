# PDMats.jl

Uniform interface for positive definite matrices of various structures.

[![Build Status](https://travis-ci.org/JuliaStats/PDMats.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/PDMats.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaStats/PDMats.jl.svg)](https://coveralls.io/r/JuliaStats/PDMats.jl?branch=master)
[![PDMats](http://pkg.julialang.org/badges/PDMats_0.5.svg)](http://pkg.julialang.org/?pkg=PDMats&ver=0.5)
[![PDMats](http://pkg.julialang.org/badges/PDMats_0.6.svg)](http://pkg.julialang.org/?pkg=PDMats&ver=0.6)

--------------

Positive definite matrices are widely used in machine learning and probabilistic modeling, especially in applications related to graph analysis and Gaussian models. It is not uncommon that positive definite matrices used in practice have special structures (e.g. diagonal), which can be exploited to accelerate computation.

*PDMats.jl* supports efficient computation on positive definite matrices of various structures. In particular, it provides uniform interfaces to use positive definite matrices of various structures for writing generic algorithms, while ensuring that the most efficient implementation is used in actual computation.

----------------


## Positive definite matrix types

This package defines an abstract type `AbstractPDMat{T<:Real}` as the base type for positive definite matrices with different internal representations.

Elemenent types are in princple all Real types, but in practice this is limited by the support for floating point types in Base.LinAlg.Cholesky.
  - `Float64`     Fully supported from Julia 0.3.
  - `Float32`     Fully supported from Julia 0.4.2. Full, diagonal and scale matrix types are supported in Julia 0.3 or higher.
  - `Float16`     Promoted to `Float32` for full, diagonal and scale matrix. Currently unsupported for sparse matrix.
  - `BigFloat`    Supported in Julia 0.4 for full, diagonal and scale matrix. Currently unsupported for sparse matrix.

* `PDMat`: full covariance matrix, defined as

```julia
struct PDMat{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int                    # matrix dimension
    mat::S                      # input matrix
    chol::Cholesky{T,S}         # Cholesky factorization of mat
end

# Constructors

PDMat(mat, chol)    # with both the input matrix and a Cholesky factorization

PDMat(mat)          # with the input matrix, of type Matrix or Symmetric
                    # Remarks: the Cholesky factorization will be computed
                    # upon construction.

PDMat(chol)         # with the Cholesky factorization
                    # Remarks: the full matrix will be computed upon
                    # construction.
```


* `PDiagMat`: diagonal matrix, defined as

```julia
struct PDiagMat{T<:Real,V<:AbstractVector} <: AbstractPDMat{T}
    dim::Int                    # matrix dimension
    diag::V                     # the vector of diagonal elements
    inv_diag::V                 # the element-wise inverse of diag
end

# Constructors

PDiagMat(v,inv_v)   # with the vector of diagonal elements and its inverse
PDiagMat(v)         # with the vector of diagonal elements
                    # inv_diag will be computed upon construction
```


* `ScalMat`: uniform scaling matrix, as `v * eye(d)`, defined as

```julia
struct ScalMat{T<:Real} <: AbstractPDMat{T}
    dim::Int         # matrix dimension
    value::T         # diagonal value (shared by all diagonal elements)
    inv_value::T     # inv(value)
end


# Constructors

ScalMat(d, v, inv_v) # with dimension d, diagonal value v and its inverse inv_v
ScalMat(d, v)        # with dimension d and diagonal value v
```


* `PDSparseMat`: sparse covariance matrix, defined as

```julia
struct PDSparseMat{T<:Real,S<:AbstractSparseMatrix} <: AbstractPDMat{T}
    dim::Int                       # matrix dimension
    mat::SparseMatrixCSC           # input matrix
    chol::CholTypeSparse           # Cholesky factorization of mat
end

# Constructors

PDSparseMat(mat, chol)    # with both the input matrix and a Cholesky factorization

PDSparseMat(mat)          # with the sparse input matrix, of type SparseMatrixCSC
                          # Remarks: the Cholesky factorization will be computed
                          # upon construction.

PDSparseMat(chol)         # with the Cholesky factorization
                          # Remarks: the sparse matrix 'mat' will be computed upon
                          # construction.
```


## Common interface

All subtypes of `AbstractPDMat` share the same API, *i.e.* with the same set of methods to operate on their instances. These methods are introduced below, where `a` is an instance of a subtype of `AbstractPDMat` to represent a positive definite matrix, `x` be a column vector or a matrix with `size(x,1) == dim(a)`, and `c` be a positive real value.

```julia

dim(a)      # return the dimension of `a`.
            # Let `a` represent a d x d matrix, then `dim(a)` returns d.

size(a)     # return the size tuple of `a`, i.e. `(dim(a), dim(a))`.

size(a, i)  # return the i-th dimension of `a`.

ndims(a)    # the number of dimensions, which is always 2.

eltype(a)   # the element type

Matrix(a)   # return a copy of the matrix in full form.

diag(a)     # return a vector of diagonal elements.

inv(a)      # inverse of `a`, of a proper subtype of `AbstractPDMat`.
            # Note: when `a` is an instance of either `PDMat`, `PDiagMat`,
            # and `ScalMat`, `inv(a)` is of the same type of `a`.
            # This needs not be required for customized subtypes -- the
            # inverse does not always has the same pattern as `a`.

eigmax(a)   # maximum eigenvalue of `a`.

eigmin(a)   # minimum eigenvalue of `a`.

logdet(a)   # log-determinant of `a`, computed in a numerically stable way.

a * x       # multiple `a` with `x` (forward transform)

a \ x       # multiply `inv(a)` with `x` (backward transform).
            # The internal implementation may not explicitly instantiate
            # the inverse of `a`.

a * c       # scale `a` by a positive scale `c`.
            # The result is in general of the same type of `a`.

c * a       # equivalent to a * c.

a + b       # add two positive definite matrices

pdadd(a, b, c)      # add `a` with `b * c`, where both `a` and `b` are
                    # instances of `AbstractPDMat`.

pdadd(m, a)         # add `a` to a dense matrix `m` of the same size.

pdadd(m, a, c)      # add `a * c` to a dense matrix `m` of the same size.

pdadd!(m, a)        # add `a` to a dense matrix `m` of the same size inplace.

pdadd!(m, a, c)     # add `a * c` to a dense matrix `m` of the same size,
                    # inplace.

pdadd!(r, m, a)     # add `a` to a dense matrix `m` of the same size, and write
                    # the result to `r`.

pdadd!(r, m, a, c)  # add `a * c` to a dense matrix `m` of the same size, and
                    # write the result to `r`.

quad(a, x)          # compute x' * a * x when `x` is a vector.
                    # perform such computation in a column-wise fashion, when
                    # `x` is a matrix, and return a vector of length `n`,
                    # where `n` is the number of columns in `x`.

quad!(r, a, x)      # compute x' * a * x in a column-wise fashion, and write
                    # the results to `r`.

invquad(a, x)       # compute x' * inv(a) * x when `x` is a vector.
                    # perform such computation in a column-wise fashion, when
                    # `x` is a matrix, and return a vector of length `n`.

invquad!(r, a, x)   # compute x' * inv(a) * x in a column-wise fashion, and
                    # write the results to `r`.

X_A_Xt(a, x)        # compute `x * a * x'` for a matrix `x`.

Xt_A_X(a, x)        # compute `x' * a * x` for a matrix `x`.

X_invA_Xt(a, x)     # compute `x * inv(a) * x'` for a matrix `x`.

Xt_invA_X(a, x)     # compute `x' * inv(a) * x` for a matrix `x`.

whiten(a, x)        # whitening transform. `x` can be a vector or a matrix.
                    #
                    # Note: If the covariance of `x` is `a`, then the
                    # covariance of the transformed result is an identity
                    # matrix.

whiten!(a, x)       # whitening transform inplace, directly updating `x`.

whiten!(r, a, x)    # write the transformed result to `r`.

unwhiten(a, x)      # inverse of whitening transform. `x` can be a vector or
                    # a matrix.
                    #
                    # Note: If the covariance of `x` is an identity matrix,
                    # then the covariance of the transformed result is `a`.
                    # Note: the un-whitening transform is useful for
                    # generating Gaussian samples.

unwhiten!(a, x)     # un-whitening transform inplace, updating `x`.

unwhiten!(r, a, x)  # write the transformed result to `r`.

test_pdmat(a, amat)     # test the correctness of implementation, given an
                        # instance of some sub-type of `AbstractPDMat`, and
                        # a corresponding full matrix.
                        #
                        # Note: this function is provided for the developers
                        # who want to define their own customized sub types.
```


## Define Customized Subtypes

In some situation, it is useful to define a customized subtype of `AbstractPDMat` to capture positive definite matrices with special structures. For this purpose, one has to define a subset of methods (as listed below), and other methods will be automatically provided.

```julia

# Let `M` be the name of the subtype, then the following methods need
# to be implemented for `M`:

dim(a::M)       # return the dimension of `a`

Matrix(a::M)    # return a copy of the matrix in full form, of type
                # `Matrix{eltype(M)}`.

diag(a::M)      # return the vector of diagonal elements, of type
                # `Vector{eltype(M)}`.

pdadd!(m, a, c)     # add `a * c` to a dense matrix `m` of the same size
                    # inplace.

* (a::M, c::Real)        # return a scaled version of `a`.

* (a::M, x::DenseVecOrMat)  # transform `x`, i.e. compute `a * x`.

\ (a::M, x::DenseVecOrMat)  # inverse transform `x`, i.e. compute `inv(a) * x`.

inv(a::M)       # compute the inverse of `a`.

logdet(a::M)    # compute the log-determinant of `a`.

eigmax(a::M)    # compute the maximum eigenvalue of `a`.

eigmin(a::M)    # compute the minimum eigenvalue of `a`.

whiten!(r::DenseVecOrMat, a::M, x::DenseVecOrMat)  # whitening transform,
                                                   # write result to `r`.

unwhiten!(r::DenseVecOrMat, a::M, x::DenseVecOrMat)  # un-whitening transform,
                                                     # write result to `r`.

quad(a::M, x::DenseVector)      # compute `x' * a * x`

quad!(r::AbstractArray, a::M, x::DenseMatrix)   # compute `x' * a * x` in
                                                # a column-wise manner

invquad(a::M, x::DenseVector)   # compute `x' * inv(a) * x`

invquad!(r::AbstractArray, a::M, x::DenseMatrix) # compute `x' * inv(a) * x`
                                                 # in a column-wise manner

X_A_Xt(a::M, x::DenseMatrix)        # compute `x * a * x'`

Xt_A_X(a::M, x::DenseMatrix)        # compute `x' * a * x`

X_invA_Xt(a::M, x::DenseMatrix)     # compute `x * inv(a) * x'`

Xt_invA_X(a::M, x::DenseMatrix)     # compute `x' * inv(a) * x`
```
