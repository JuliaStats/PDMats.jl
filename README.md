# PDMats.jl

Uniform interface for positive definite matrices of various structures. 
[![Build Status](https://travis-ci.org/JuliaStats/PDMats.jl.png?branch=master)](https://travis-ci.org/JuliaStats/PDMats.jl)

--------------

Positive definite matrices are widely used in machine learning and probabilistic modeling, especially in applications related to graph analysis and Gaussian models. It is not uncommon that positive definite matrices used in practice have special structures (e.g. diagonal), which can be exploited to accelerate computation. 

*PDMats.jl* supports efficient computation on positive definite matrices of various structures. In particular, it provides uniform interfaces to use positive definite matrices of various structures for writing generic algorithms, while ensuring that the most efficient implementation is used in actual computation.

----------------


## Positive definite matrix types

This package defines an abstract type ``AbstractPDMat`` as the base type for positive definite matrices with different internal representations. 

* ``PDMat``: full covariance matrix, defined as

```julia
immutable PDMat <: AbstractPDMat
    dim::Int                    # matrix dimension
    mat::Matrix{Float64}        # input matrix
    chol::Cholesky{Float64}     # Cholesky factorization of mat
end

# Constructors

PDMat(mat, chol)    # with both the input matrix and a Cholesky factorization

PDMat(mat)          # with the input matrix, of type Matrix or Symmetric
                    # Remarks: the Cholesky factorization will be computed
                    # upon construction.

PDMat(mat, uplo)    # with the input matrix, and an uplo argument (:U or :L)
                    # to specify the way Choleksy is done

PDMat(chol)         # with the Cholesky factorization
                    # Remarks: the full matrix will be computed upon 
                    # construction.
```


* ``PDiagMat``: diagonal matrix, defined as

```julia
immutable PDiagMat <: AbstractPDMat
    dim::Int                    # matrix dimension
    diag::Vector{Float64}       # the vector of diagonal elements
    inv_diag::Vector{Float64}   # the element-wise inverse of diag
end

# Constructors

PDiagMat(v)     # with the vector of diagonal elements
                # inv_diag will be computed upon construction
```


* ``ScalMat``: uniform scaling matrix, as ``v * eye(d)``, defined as

```julia
immutable ScalMat <: AbstractPDMat
    dim::Int                # matrix dimension
    value::Float64          # diagonal value (shared by all diagonal elements)
    inv_value::Float64      # inv(value)
end

# Constructors

ScalMat(d, v)       # with dimension d and diagonal value v
```


## Common interface

All subtypes of ``AbstractPDMat`` share the same API, *i.e.* with the same set of methods to operate on their instances. These methods are introduced below, where ``a`` is an instance of a subtype of ``AbstractPDMat`` to represent a positive definite matrix, ``x`` be a column vector or a matrix with ``size(x,1) == dim(a)``, and ``c`` be a positive real value.

```julia

dim(a)      # return the dimension of `a`. 
            # Let `a` represent a d x d matrix, then `dim(a)` returns d.

size(a)     # return the size tuple of `a`, i.e. `(dim(a), dim(a))`.

size(a, i)  # return the i-th dimension of `a`.

ndims(a)    # the number of dimensions, which is always 2.

eltype(a)   # the element type, which is always `Float64`

full(a)     # return a copy of the matrix in full form.

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


```



