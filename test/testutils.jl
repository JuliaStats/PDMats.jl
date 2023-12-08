# Utilities for testing
#
#       One can use the facilities provided here to simplify the testing of
#       the implementation of a subtype of AbstractPDMat
#

using PDMats, SuiteSparse, Test, Random

Random.seed!(10)

const HAVE_CHOLMOD = isdefined(SuiteSparse, :CHOLMOD)
const PDMatType = HAVE_CHOLMOD ? Union{PDMat, PDSparseMat, PDiagMat} : Union{PDMat, PDiagMat}

## driver function
function test_pdmat(C, Cmat::Matrix;
                    verbose::Int=2,             # the level to display intermediate steps
                    cmat_eq::Bool=false,        # require Cmat and Matrix(C) to be exactly equal
                    t_diag::Bool=true,          # whether to test diag method
                    t_cholesky::Bool=true,      # whether to test cholesky method
                    t_scale::Bool=true,         # whether to test scaling
                    t_add::Bool=true,           # whether to test pdadd
                    t_det::Bool=true,           # whether to test det method
                    t_logdet::Bool=true,        # whether to test logdet method
                    t_eig::Bool=true,           # whether to test eigmax and eigmin
                    t_mul::Bool=true,           # whether to test multiplication
                    t_div::Bool=true,           # whether to test division
                    t_quad::Bool=true,          # whether to test quad & invquad
                    t_triprod::Bool=true,       # whether to test X_A_Xt, Xt_A_X, X_invA_Xt, and Xt_invA_X
                    t_whiten::Bool=true         # whether to test whiten and unwhiten
                    )

    d = size(Cmat, 1)
    verbose >= 1 && printstyled("Testing $(typeof(C)) of size ($d, $d)\n", color=:blue)

    pdtest_basics(C, Cmat, d, verbose)
    pdtest_cmat(C, Cmat, cmat_eq, verbose)

    t_diag && pdtest_diag(C, Cmat, cmat_eq, verbose)
    isa(C, PDMatType) && t_cholesky && pdtest_cholesky(C, Cmat, cmat_eq, verbose)
    t_scale && pdtest_scale(C, Cmat, verbose)
    t_add && pdtest_add(C, Cmat, verbose)
    t_det && pdtest_det(C, Cmat, verbose)
    t_logdet && pdtest_logdet(C, Cmat, verbose)

    t_eig && pdtest_eig(C, Cmat, verbose)
    Imat = inv(Cmat)

    n = 5
    X = rand(eltype(C),d,n) .- convert(eltype(C),0.5)

    t_mul && pdtest_mul(C, Cmat, X, verbose)
    t_div && pdtest_div(C, Imat, X, verbose)
    t_quad && pdtest_quad(C, Cmat, Imat, X, verbose)
    t_triprod && pdtest_triprod(C, Cmat, Imat, X, verbose)

    t_whiten && pdtest_whiten(C, Cmat, verbose)

    verbose >= 2 && println()
end


## core testing functions

_pdt(vb::Int, s) = (vb >= 2 && printstyled("    .. testing $s\n", color=:green))


function pdtest_basics(C, Cmat::Matrix, d::Int, verbose::Int)
    _pdt(verbose, "dim")
    @test @test_deprecated(dim(C)) == d

    _pdt(verbose, "size")
    @test size(C) == (d, d)
    @test size(C, 1) == d
    @test size(C, 2) == d
    @test size(C, 3) == 1

    _pdt(verbose, "ndims")
    @test ndims(C) == 2

    _pdt(verbose, "length")
    @test length(C) == d * d

    _pdt(verbose, "eltype")
    @test eltype(C) == eltype(Cmat)
#    @test eltype(typeof(C)) == eltype(typeof(Cmat))

    _pdt(verbose, "index")
    @test all(C[i] == Cmat[i] for i in 1:(d^2))
    @test all(C[i, j] == Cmat[i, j] for j in 1:d, i in 1:d)

    _pdt(verbose, "isposdef")
    @test isposdef(C)

    _pdt(verbose, "ishermitian")
    @test ishermitian(C)

    _pdt(verbose, "AbstractPDMat")
    M = AbstractPDMat(C)
    @test M isa AbstractPDMat
    if C isa AbstractPDMat
        @test M === C
    end

    _pdt(verbose, "Matrix")
    M = Matrix(C)
    @test M isa Matrix
    @test M == Cmat
end


function pdtest_cmat(C, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "full")
    if cmat_eq
        @test Matrix(C) == Cmat
    else
        @test Matrix(C) ≈ Cmat
    end
end


function pdtest_diag(C, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "diag")
    if cmat_eq
        @test diag(C) == diag(Cmat)
    else
        @test diag(C) ≈ diag(Cmat)
    end
end

function pdtest_cholesky(C::Union{PDMat, PDiagMat, ScalMat}, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "cholesky")
    if cmat_eq
        @test cholesky(C).U == cholesky(Cmat).U
    else
        @test cholesky(C).U ≈ cholesky(Cmat).U
    end
    # regression test: https://github.com/JuliaStats/PDMats.jl/pull/182
    if C isa Union{PDiagMat, ScalMat}
        size_of_sqrt_diag = C.dim * sizeof(float(eltype(C)))
        # allow some overhead for wrapper types
        max_allocations = max(1.05 * size_of_sqrt_diag, 128 + size_of_sqrt_diag)
        @test (@allocated cholesky(C)) <= max_allocations
    end
end

if HAVE_CHOLMOD
    function pdtest_cholesky(C::PDSparseMat, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
        _pdt(verbose, "cholesky")
        # We special case PDSparseMat because we can't perform equality checks on
        # `SuiteSparse.CHOLMOD.Factor`s and `SuiteSparse.CHOLMOD.FactorComponent`s
        @test diag(cholesky(C)) ≈ diag(cholesky(Cmat).U)
        # NOTE: `==` also doesn't work because `diag(cholesky(C))` will return `Vector{Float64}`
        # even if the inputs are `Float32`s.
    end
end

function pdtest_scale(C, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "scale")
    @test Matrix(C * convert(eltype(C),2)) ≈ Cmat * convert(eltype(C),2)
    @test Matrix(convert(eltype(C),2) * C) ≈ convert(eltype(C),2) * Cmat
end


function pdtest_add(C, Cmat::Matrix, verbose::Int)
    M = rand(eltype(C),size(Cmat))
    _pdt(verbose, "add")
    @test C + M ≈ Cmat + M
    @test M + C ≈ M + Cmat

    _pdt(verbose, "add_scal")
    @test pdadd(M, C, convert(eltype(C),2)) ≈ M + Cmat * convert(eltype(C),2)

    _pdt(verbose, "add_scal!")
    R = M + Cmat * convert(eltype(C),2)
    Mr = pdadd!(M, C, convert(eltype(C),2))
    @test Mr === M
    @test Mr ≈ R
end

function pdtest_det(C, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "det")
    @test det(C) ≈ det(Cmat)

    # generic fallback in LinearAlgebra performs LU decomposition
    if C isa Union{PDMat,PDiagMat,ScalMat}
	@test iszero(@allocated det(C))
    end
end

function pdtest_logdet(C, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "logdet")
    @test logdet(C) ≈ logdet(Cmat)

    # generic fallback in LinearAlgebra performs LU decomposition
    if C isa Union{PDMat,PDiagMat,ScalMat}
	@test iszero(@allocated logdet(C))
    end
end


function pdtest_eig(C, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "eigmax")
    @test eigmax(C) ≈ eigmax(Cmat)

    _pdt(verbose, "eigmin")
    @test eigmin(C) ≈ eigmin(Cmat)
end


function pdtest_mul(C, Cmat::Matrix, verbose::Int)
    n = 5
    X = rand(eltype(C), size(C, 1), n)
    pdtest_mul(C, Cmat, X, verbose)
end


function pdtest_mul(C, Cmat::Matrix, X::Matrix, verbose::Int)
    _pdt(verbose, "multiply")
    d, n = size(X)
    @assert d == size(C, 1) == size(C, 2)
    @assert size(Cmat) == size(C)
    @test C * X ≈ Cmat * X

    y = similar(C * X, d)
    ymat = similar(Cmat * X, d)
    for i = 1:n
        xi = vec(copy(X[:,i]))
        @test C * xi ≈ Cmat * xi

        mul!(y, C, xi)
        mul!(ymat, Cmat, xi)
        @test y ≈ ymat
    end

    # Dimension mismatches
    @test_throws DimensionMismatch C * rand(d + 1)
    @test_throws DimensionMismatch C * rand(d + 1, n)
end


function pdtest_div(C, Imat::Matrix, X::Matrix, verbose::Int)
    _pdt(verbose, "divide")
    d, n = size(X)
    @assert d == size(C, 1) == size(C, 2)
    @assert size(Imat) == size(C)
    @test C \ X ≈ Imat * X
    # Right division with Choleskyrequires https://github.com/JuliaLang/julia/pull/32594
    # CHOLMOD throws error since no method is found for
    # `rdiv!(::Matrix{Float64}, ::SuiteSparse.CHOLMOD.Factor{Float64})`
    check_rdiv = !(C isa PDMat && VERSION < v"1.3.0-DEV.562") && !(C isa PDSparseMat && HAVE_CHOLMOD)
    check_rdiv && @test Matrix(X') / C ≈ (C \ X)'

    for i = 1:n
        xi = vec(copy(X[:,i]))
        @test C \ xi ≈ Imat * xi
        check_rdiv && @test Matrix(xi') / C ≈ (C \ xi)'
    end


    # Dimension mismatches
    @test_throws DimensionMismatch C \ rand(d + 1)
    @test_throws DimensionMismatch C \ rand(d + 1, n)
    if check_rdiv
        @test_throws DimensionMismatch rand(1, d + 1) / C
        @test_throws DimensionMismatch rand(n, d + 1) / C
    end
end


function pdtest_quad(C, Cmat::Matrix, Imat::Matrix, X::Matrix, verbose::Int)
    n = size(X, 2)

    _pdt(verbose, "quad")
    r_quad = zeros(eltype(C),n)
    for i = 1:n
        xi = vec(X[:,i])
        r_quad[i] = dot(xi, Cmat * xi)
        @test quad(C, xi) ≈ r_quad[i]
        @test quad(C, view(X,:,i)) ≈ r_quad[i]
    end
    @test quad(C, X) ≈ r_quad
    r = similar(r_quad)
    @test quad!(r, C, X) === r
    @test r ≈ r_quad

    _pdt(verbose, "invquad")
    r_invquad = zeros(eltype(C),n)
    for i = 1:n
        xi = vec(X[:,i])
        r_invquad[i] = dot(xi, Imat * xi)
        @test invquad(C, xi) ≈ r_invquad[i]
        @test invquad(C, view(X,:,i)) ≈ r_invquad[i]
    end
    @test invquad(C, X) ≈ r_invquad
    r = similar(r_invquad)
    @test invquad!(r, C, X) === r
    @test r ≈ r_invquad
end


function pdtest_triprod(C, Cmat::Matrix, Imat::Matrix, X::Matrix, verbose::Int)
    d, n = size(X)
    @assert d == size(C, 1) == size(C, 2)
    Xt = copy(transpose(X))

    _pdt(verbose, "X_A_Xt")
    M = X_A_Xt(C, Xt)
    @test M ≈ Xt * Cmat * X
    @test issymmetric(M)
    @test_throws DimensionMismatch X_A_Xt(C, rand(n, d + 1))

    _pdt(verbose, "Xt_A_X")
    M = Xt_A_X(C, X)
    @test M ≈ Xt * Cmat * X
    @test issymmetric(M)
    @test_throws DimensionMismatch Xt_A_X(C, rand(d + 1, n))

    _pdt(verbose, "X_invA_Xt")
    M = X_invA_Xt(C, Xt)
    @test M ≈ Xt * Imat * X
    @test issymmetric(M)
    @test_throws DimensionMismatch X_invA_Xt(C, rand(n, d + 1))

    _pdt(verbose, "Xt_invA_X")
    M = Xt_invA_X(C, X)
    @test M ≈ Xt * Imat * X
    @test issymmetric(M)
    @test_throws DimensionMismatch Xt_invA_X(C, rand(d + 1, n))
end


function pdtest_whiten(C, Cmat::Matrix, verbose::Int)
    Y = PDMats.chol_lower(Cmat)
    Q = qr(convert(Array{eltype(C),2},randn(size(Cmat)))).Q
    Y = Y * Q'                    # generate a matrix Y such that Y * Y' = C
    @test Y * Y' ≈ Cmat
    d = size(C, 1)

    _pdt(verbose, "whiten")
    Z = whiten(C, Y)
    @test Z * Z' ≈ Matrix{eltype(C)}(I, d, d)
    for i = 1:d
        @test whiten(C, Y[:,i]) ≈ Z[:,i]
    end

    _pdt(verbose, "whiten!")
    Z2 = copy(Y)
    whiten!(C, Z2)
    @test Z ≈ Z2

    _pdt(verbose, "unwhiten")
    X = unwhiten(C, Z)
    @test X * X' ≈ Cmat
    for i = 1:d
        @test unwhiten(C, Z[:,i]) ≈ X[:,i]
    end

    _pdt(verbose, "unwhiten!")
    X2 = copy(Z)
    unwhiten!(C, X2)
    @test X ≈ X2

    _pdt(verbose, "whiten-unwhiten")
    @test unwhiten(C, whiten(C, Matrix{eltype(C)}(I, d, d))) ≈ Matrix{eltype(C)}(I, d, d)
    @test whiten(C, unwhiten(C, Matrix{eltype(C)}(I, d, d))) ≈ Matrix{eltype(C)}(I, d, d)
end


# testing functions for kron and sqrt

_randPDMat(T, n) = (X = randn(T, n, n); PDMat(X * X' + LinearAlgebra.I))
_randPDiagMat(T, n) = PDiagMat(rand(T, n))
_randScalMat(T, n) = ScalMat(n, rand(T))
_randPDSparseMat(T, n) = (X = T.(sprand(n, 1, 0.5)); PDSparseMat(X * X' + LinearAlgebra.I))

function _pd_compare(A::AbstractPDMat, B::AbstractPDMat)
    @test size(A) == size(B)
    @test Matrix(A) ≈ Matrix(B)
    @test cholesky(A).L ≈ cholesky(B).L
    @test cholesky(A).U ≈ cholesky(B).U
end
