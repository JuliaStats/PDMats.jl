# PDMats.jl

Uniform interface for positive definite matrices of various structures. 
[![Build Status](https://travis-ci.org/JuliaStats/PDMats.jl.png?branch=master)](https://travis-ci.org/JuliaStats/PDMats.jl)

--------------

Positive definite matrices are widely used in machine learning and probabilistic modeling, especially in applications related to graph analysis and Gaussian models. It is not uncommon that positive definite matrices used in practice have special structures (e.g. diagonal), which can be exploited to accelerate computation. 

*PDMats.jl* supports efficient computation on positive definite matrices of various structures. In particular, it provides uniform interfaces to use positive definite matrices of various structures for writing generic algorithms, while ensuring that the most efficient implementation is used in actual computation.

----------------


## Positive definite matrix types

This package defines an abstract type ``AbstractPDMat`` to capture positive definite matrices of various structures, as well as three concrete sub-types: ``PDMat``, ``PDiagMat``, ``ScalMat``, which can be constructed as follows

* ``PDMat``: representing a normal positive definite matrix in its full matrix form. **Construction:** ``PDMat(C)``.

* ``PDiagMat``: representing a positive diagonal matrix. **Construction:** ``PDiagMat(v)``, where ``v`` is the vector of diagonal elements.

* ``ScalMat(d, v)``: representing a scaling matrix of the form ``v * eye(d)``. **Construction:** ``ScalMat(d, v)``, where ``d`` is the matrix dimension (the size of the matrix is ``d x d``), and ``v`` is a scalar value.

**Notes:** Compact representation is used internally. For example, an instance of ``PDiagMat`` only contains a vector of diagonal elements instead of the full diagonal matrix, and ``ScalMat`` only contains a scalar value. While, for ``PDMat``, a Cholesky factorization is computed and contained in the instance for efficient computation.


## Common interface

Functions are defined to operate on positive definite matrices through a uniform interface. In the description below, We let ``a`` be a positive definite matrix, *i.e* an instance of a subtype of ``AbstractPDMat``, ``x`` be a column vector or a matrix, and ``c`` be a positive scalar. 

* **dim**(a)

   Return the dimension of the matrix. If it is a ``d x d`` matrix, this returns ``d``.

* **full**(a)

    Return a copy of the matrix in full form.

* **logdet**(a)

    Return the log-determinant of the matrix.

* **diag**(a)

    Return a vector of diangonal elements.

* a * x

    Perform matrix-vector/matrix-matrix multiplication. 

* a \ x

    Solve linear equation, equivalent to ``inv(a) * x``, but implemented in a more efficient way.

* a * c, c * a

    Scalar product, multiply ``a`` with a scalar ``c``.

* **unwhiten**(a, x)   

    Unwhitening transform. 

    If ``x`` satisfies the standard Gaussian distribution, then ``unwhiten(a, x)`` has a distribution 
    of covariance ``a``.

* **whiten**(a, x)

    Whitening transform.

    If ``x`` satisfies a distributiion of covariance ``a``, then the covariance of ``whiten(a, x)`` is the identity matrix. 

    **Note:** ``whiten`` and ``unwhiten`` are mutually inverse operations.

* **unwhiten!**(a, x)

    Inplace unwhitening, ``x`` will be updated.

* **whiten!**(a, x)

    Inplace whitening, ``x`` will be updated.

* **quad**(a, x)

    Compute ``x' * a * x`` in an efficient way. Here, ``x`` can be a vector or a matrix.

    If ``x`` is a vector, it returns a scalar value.
    If ``x`` is a matrix, is performs column-wise computation and returns a vector ``r``, 
    such that ``r[i]`` is ``x[:,i]' * a * x[:,i]``.

* **invquad**(a, x)

    Compute ``x' * inv(a) * x`` in an efficient way (without computing ``inv(a)``). 
    Here, ``x`` can be a vector or a matrix (for column-wise computation).

* **quad!**(r, a, x)

    Inplace column-wise computation of ``quad`` on a matrix ``x``.

* **invquad!**(r, a, x)

    Inplace column-wise computation of ``invquad`` on a matrix ``x``.

* **X_A_Xt**(a, x)

    Computes ``x * a * x'`` for matrix ``x``.

* **Xt_A_X**(a, x)

    Computes ``x' * a * x`` for matrix ``x``.

* **X_invA_Xt**(a, x)

    Computes ``x * inv(a) * x'`` for matrix ``x``.

* **Xt_invA_X**(a, x)

    Computes ``x' * inv(a) * x`` for matrix ``x``.

* a1 + a2

    Add two positive definite matrices (promoted to a proper type).

* a + x

    Add a positive definite matrix and an ordinary square matrix (returns an ordinary matrix).

* **add!**(x, a)

    Add the positive definite matrix ``a`` to an ordinary matrix ``m`` (inplace).

* **add_scal!**(x, a, c)

    Add ``a * c`` to an ordinary matrix ``x`` (inplace).

* **add_scal**(a1, a2, c)

    Return ``a1 + a2 * c`` (promoted to a proper type).

**Note:** Specialized version of each of these functions are implemented for each specific postive matrix types using the most efficient routine (depending on the corresponding structures.)

