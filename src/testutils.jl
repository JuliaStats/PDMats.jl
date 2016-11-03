# Utilities for testing
#
#       One can use the facilities provided here to simplify the testing of
#       the implementation of a subtype of AbstractPDMat
#

import Base.Test: @test, @test_approx_eq

## driver function
function test_pdmat(C::AbstractPDMat, Cmat::Matrix;
                    verbose::Int=2,             # the level to display intermediate steps
                    cmat_eq::Bool=false,        # require Cmat and full(C) to be exactly equal
                    t_diag::Bool=true,          # whethet to test diag method
                    t_scale::Bool=true,         # whether to test scaling
                    t_add::Bool=true,           # whether to test pdadd
                    t_logdet::Bool=true,        # whether to test logdet method
                    t_eig::Bool=true,           # whether to test eigmax and eigmin
                    t_mul::Bool=true,           # whether to test multiplication
                    t_rdiv::Bool=true,          # whether to test right division (solve)
                    t_quad::Bool=true,          # whether to test quad & invquad
                    t_triprod::Bool=true,       # whether to test X_A_Xt, Xt_A_X, X_invA_Xt, and Xt_invA_X
                    t_whiten::Bool=true         # whether to test whiten and unwhiten
                    )

    d = size(Cmat, 1)
    verbose >= 1 && print_with_color(:blue, "Testing $(typeof(C)) with dim = $d\n")

    pdtest_basics(C, Cmat, d, verbose)
    pdtest_cmat(C, Cmat, cmat_eq, verbose)

    t_diag && pdtest_diag(C, Cmat, cmat_eq, verbose)
    t_scale && pdtest_scale(C, Cmat, verbose)
    t_add && pdtest_add(C, Cmat, verbose)
    t_logdet && pdtest_logdet(C, Cmat, verbose)

    t_eig && pdtest_eig(C, Cmat, verbose)
    Imat = inv(Cmat)

    n = 5
    X = rand(eltype(C),d,n) - convert(eltype(C),0.5)

    t_mul && pdtest_mul(C, Cmat, X, verbose)
    t_rdiv && pdtest_rdiv(C, Imat, X, verbose)
    t_quad && pdtest_quad(C, Cmat, Imat, X, verbose)
    t_triprod && pdtest_triprod(C, Cmat, Imat, X, verbose)

    t_whiten && pdtest_whiten(C, Cmat, verbose)

    verbose >= 2 && println()
end


## core testing functions

_pdt(vb::Int, s) = (vb >= 2 && print_with_color(:green, "    .. testing $s\n"))


function pdtest_basics(C::AbstractPDMat, Cmat::Matrix, d::Int, verbose::Int)
    _pdt(verbose, "dim")
    @test dim(C) == d

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
end


function pdtest_cmat(C::AbstractPDMat, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "full")
    if cmat_eq
        @test full(C) == Cmat
    else
        @test_approx_eq full(C) Cmat
    end
end


function pdtest_diag(C::AbstractPDMat, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "diag")
    if cmat_eq
        @test diag(C) == diag(Cmat)
    else
        @test_approx_eq diag(C) diag(Cmat)
    end
end


function pdtest_scale(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "scale")
    @test_approx_eq full(C * convert(eltype(C),2)) Cmat * convert(eltype(C),2)
    @test_approx_eq full(convert(eltype(C),2) * C) convert(eltype(C),2) * Cmat
end


function pdtest_add(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    M = rand(eltype(C),size(Cmat))
    _pdt(verbose, "add")
    @test_approx_eq C + M Cmat + M
    @test_approx_eq M + C M + Cmat

    _pdt(verbose, "add_scal")
    @test_approx_eq pdadd(M, C, convert(eltype(C),2)) M + Cmat * convert(eltype(C),2)

    _pdt(verbose, "add_scal!")
    R = M + Cmat * convert(eltype(C),2)
    Mr = pdadd!(M, C, convert(eltype(C),2))
    @test Mr === M
    @test_approx_eq Mr R
end


function pdtest_logdet(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "logdet")
    @test_approx_eq logdet(C) logdet(Cmat)
end


function pdtest_eig(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "eigmax")
    @test_approx_eq eigmax(C) eigmax(Cmat)

    _pdt(verbose, "eigmin")
    @test_approx_eq eigmin(C) eigmin(Cmat)
end


function pdtest_mul(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    n = 5
    X = rand(eltype(C),dim(C), n)

    _pdt(verbose, "multiply")
    @test_approx_eq C * X Cmat * X

    for i = 1:n
        xi = vec(copy(X[:,i]))
        @test_approx_eq C * xi Cmat * xi
    end
end


function pdtest_mul(C::AbstractPDMat, Cmat::Matrix, X::Matrix, verbose::Int)
    _pdt(verbose, "multiply")
    @test_approx_eq C * X Cmat * X

    for i = 1:size(X,2)
        xi = vec(copy(X[:,i]))
        @test_approx_eq C * xi Cmat * xi
    end
end


function pdtest_rdiv(C::AbstractPDMat, Imat::Matrix, X::Matrix, verbose::Int)
    _pdt(verbose, "rdivide")
    @test_approx_eq C \ X Imat * X

    for i = 1:size(X,2)
        xi = vec(copy(X[:,i]))
        @test_approx_eq C \ xi Imat * xi
    end
end


function pdtest_quad(C::AbstractPDMat, Cmat::Matrix, Imat::Matrix, X::Matrix, verbose::Int)
    n = size(X, 2)

    _pdt(verbose, "quad")
    r_quad = zeros(eltype(C),n)
    for i = 1:n
        xi = vec(X[:,i])
        r_quad[i] = dot(xi, Cmat * xi)
        @test_approx_eq quad(C, xi) r_quad[i]
    end
    @test_approx_eq quad(C, X) r_quad

    _pdt(verbose, "invquad")
    r_invquad = zeros(eltype(C),n)
    for i = 1:n
        xi = vec(X[:,i])
        r_invquad[i] = dot(xi, Imat * xi)
        @test_approx_eq invquad(C, xi) r_invquad[i]
    end
    @test_approx_eq invquad(C, X) r_invquad
end


function pdtest_triprod(C::AbstractPDMat, Cmat::Matrix, Imat::Matrix, X::Matrix, verbose::Int)
    Xt = copy(transpose(X))

    _pdt(verbose, "X_A_Xt")
    @test_approx_eq X_A_Xt(C, Xt) Xt * Cmat * X

    _pdt(verbose, "Xt_A_X")
    @test_approx_eq Xt_A_X(C, X) Xt * Cmat * X

    _pdt(verbose, "X_invA_Xt")
    @test_approx_eq X_invA_Xt(C, Xt) Xt * Imat * X

    _pdt(verbose, "Xt_invA_X")
    @test_approx_eq Xt_invA_X(C, X) Xt * Imat * X
end


function pdtest_whiten(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    Y = chol_lower(Cmat)
    Q = qr(convert(Array{eltype(C),2},randn(size(Cmat))))[1]
    Y = Y * Q'                    # generate a matrix Y such that Y * Y' = C
    @test_approx_eq Y * Y' Cmat
    d = dim(C)

    _pdt(verbose, "whiten")
    Z = whiten(C, Y)
    @test_approx_eq Z * Z' eye(eltype(C),d)
    for i = 1:d
        @test_approx_eq whiten(C, Y[:,i]) Z[:,i]
    end

    _pdt(verbose, "whiten!")
    Z2 = copy(Y)
    whiten!(C, Z2)
    @test_approx_eq Z Z2

    _pdt(verbose, "unwhiten")
    X = unwhiten(C, Z)
    @test_approx_eq X * X' Cmat
    for i = 1:d
        @test_approx_eq unwhiten(C, Z[:,i]) X[:,i]
    end

    _pdt(verbose, "unwhiten!")
    X2 = copy(Z)
    unwhiten!(C, X2)
    @test_approx_eq X X2

    _pdt(verbose, "whiten-unwhiten")
    @test_approx_eq unwhiten(C, whiten(C, eye(eltype(C),d))) eye(eltype(C),d)
    @test_approx_eq whiten(C, unwhiten(C, eye(eltype(C),d))) eye(eltype(C),d)
end
