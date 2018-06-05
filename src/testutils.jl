# Utilities for testing
#
#       One can use the facilities provided here to simplify the testing of
#       the implementation of a subtype of AbstractPDMat
#

using Test: @test

## driver function
function test_pdmat(C::AbstractPDMat, Cmat::Matrix;
                    verbose::Int=2,             # the level to display intermediate steps
                    cmat_eq::Bool=false,        # require Cmat and Matrix(C) to be exactly equal
                    t_diag::Bool=true,          # whether to test diag method
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
    verbose >= 1 && printstyled("Testing $(typeof(C)) with dim = $d\n", color=:blue)

    pdtest_basics(C, Cmat, d, verbose)
    pdtest_cmat(C, Cmat, cmat_eq, verbose)

    t_diag && pdtest_diag(C, Cmat, cmat_eq, verbose)
    t_scale && pdtest_scale(C, Cmat, verbose)
    t_add && pdtest_add(C, Cmat, verbose)
    t_logdet && pdtest_logdet(C, Cmat, verbose)

    t_eig && pdtest_eig(C, Cmat, verbose)
    Imat = inv(Cmat)

    n = 5
    X = rand(eltype(C),d,n) .- convert(eltype(C),0.5)

    t_mul && pdtest_mul(C, Cmat, X, verbose)
    t_rdiv && pdtest_rdiv(C, Imat, X, verbose)
    t_quad && pdtest_quad(C, Cmat, Imat, X, verbose)
    t_triprod && pdtest_triprod(C, Cmat, Imat, X, verbose)

    t_whiten && pdtest_whiten(C, Cmat, verbose)

    verbose >= 2 && println()
end


## core testing functions

_pdt(vb::Int, s) = (vb >= 2 && printstyled("    .. testing $s\n", color=:green))


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
#    @test eltype(typeof(C)) == eltype(typeof(Cmat))
end


function pdtest_cmat(C::AbstractPDMat, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "full")
    if cmat_eq
        @test Matrix(C) == Cmat
    else
        @test Matrix(C) ≈ Cmat
    end
end


function pdtest_diag(C::AbstractPDMat, Cmat::Matrix, cmat_eq::Bool, verbose::Int)
    _pdt(verbose, "diag")
    if cmat_eq
        @test diag(C) == diag(Cmat)
    else
        @test diag(C) ≈ diag(Cmat)
    end
end


function pdtest_scale(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "scale")
    @test Matrix(C * convert(eltype(C),2)) ≈ Cmat * convert(eltype(C),2)
    @test Matrix(convert(eltype(C),2) * C) ≈ convert(eltype(C),2) * Cmat
end


function pdtest_add(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
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


function pdtest_logdet(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "logdet")
    # default tolerance in isapprox is different on 0.4. rtol argument can be deleted
    # ≈ form used when 0.4 is no longer supported
    @test isapprox(logdet(C), logdet(Cmat), rtol=sqrt(max(eps(real(eltype(C))), eps(real(eltype(Cmat))))))
end


function pdtest_eig(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    _pdt(verbose, "eigmax")
    @test eigmax(C) ≈ eigmax(Cmat)

    _pdt(verbose, "eigmin")
    @test eigmin(C) ≈ eigmin(Cmat)
end


function pdtest_mul(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    n = 5
    X = rand(eltype(C),dim(C), n)

    _pdt(verbose, "multiply")
    @test C * X ≈ Cmat * X

    for i = 1:n
        xi = vec(copy(X[:,i]))
        @test C * xi ≈ Cmat * xi
    end
end


function pdtest_mul(C::AbstractPDMat, Cmat::Matrix, X::Matrix, verbose::Int)
    _pdt(verbose, "multiply")
    @test C * X ≈ Cmat * X

    for i = 1:size(X,2)
        xi = vec(copy(X[:,i]))
        @test C * xi ≈ Cmat * xi
    end
end


function pdtest_rdiv(C::AbstractPDMat, Imat::Matrix, X::Matrix, verbose::Int)
    _pdt(verbose, "rdivide")
    @test C \ X ≈ Imat * X

    for i = 1:size(X,2)
        xi = vec(copy(X[:,i]))
        @test C \ xi ≈ Imat * xi
    end
end


function pdtest_quad(C::AbstractPDMat, Cmat::Matrix, Imat::Matrix, X::Matrix, verbose::Int)
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

    _pdt(verbose, "invquad")
    r_invquad = zeros(eltype(C),n)
    for i = 1:n
        xi = vec(X[:,i])
        r_invquad[i] = dot(xi, Imat * xi)
        @test invquad(C, xi) ≈ r_invquad[i]
        @test invquad(C, view(X,:,i)) ≈ r_invquad[i]
    end
    @test invquad(C, X) ≈ r_invquad
end


function pdtest_triprod(C::AbstractPDMat, Cmat::Matrix, Imat::Matrix, X::Matrix, verbose::Int)
    Xt = copy(transpose(X))

    _pdt(verbose, "X_A_Xt")
    # default tolerance in isapprox is different on 0.4. rtol argument can be deleted
    # ≈ form used when 0.4 is no longer supported
    lhs, rhs = X_A_Xt(C, Xt), Xt * Cmat * X
    @test isapprox(lhs, rhs, rtol=sqrt(max(eps(real(float(eltype(lhs)))), eps(real(float(eltype(rhs)))))))

    _pdt(verbose, "Xt_A_X")
    lhs, rhs = Xt_A_X(C, X), Xt * Cmat * X
    @test isapprox(lhs, rhs, rtol=sqrt(max(eps(real(float(eltype(lhs)))), eps(real(float(eltype(rhs)))))))

    _pdt(verbose, "X_invA_Xt")
    @test X_invA_Xt(C, Xt) ≈ Xt * Imat * X

    _pdt(verbose, "Xt_invA_X")
    @test Xt_invA_X(C, X) ≈ Xt * Imat * X
end


function pdtest_whiten(C::AbstractPDMat, Cmat::Matrix, verbose::Int)
    Y = chol_lower(Cmat)
    Q = qr(convert(Array{eltype(C),2},randn(size(Cmat)))).Q
    Y = Y * Q'                    # generate a matrix Y such that Y * Y' = C
    @test Y * Y' ≈ Cmat
    d = dim(C)

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
