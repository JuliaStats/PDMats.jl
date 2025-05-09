# Congruent transforms X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X for guaranteed PD return values

for f in (:X_A_Xt, :Xt_A_X)
    @eval begin
        function $(f)(A::ScalMat, B::ScalMat)
            @check_argdims A.dim == B.dim
            return ScalMat(A.dim, abs2(B.value) * A.value)
        end
        $(f)(A::PDiagMat, B::PDiagMat) = PDiagMat(abs2.(B.diag) .* A.diag)
        $(f)(A::PDiagMat, B::ScalMat) = PDiagMat(abs2(B.value) .* A.diag)
        function $(f)(A::ScalMat, B::PDiagMat)
            @check_argdims A.dim == size(B, 1)
            return PDiagMat(abs2.(B.diag) .* A.value)
        end
        function $(f)(A::PDMat, B::ScalMat)
            @check_argdims B.dim == size(A, 1)
            b2 = abs2(B.value)
            mat = b2 * A.mat
            uplo = A.chol.uplo
            if uplo === 'U'
                factors = A.chol.factors * B.value'
            else
                factors = B.value * A.chol.factors
            end
            chol = Cholesky(factors, uplo, A.chol.info)
            return PDMat(mat, chol)
        end
        function $(f)(A::PDMat, B::PDiagMat)
            b = B.diag
            mat = A.mat .* (b .* b')
            uplo = A.chol.uplo
            if uplo === 'U'
                factors = A.chol.factors .* b'
            else
                factors = b .* A.chol.factors
            end
            chol = Cholesky(factors, uplo, A.chol.info)
            return PDMat(mat, chol)
        end
    end
end

for f in (:X_invA_Xt, :Xt_invA_X)
    @eval begin
        $(f)(A::ScalMat, B::ScalMat) = B * (A \ B)
        $(f)(A::PDiagMat, B::PDiagMat) = PDiagMat(B.diag .* (A.diag .\ B.diag))
        $(f)(A::PDiagMat, B::ScalMat) = PDiagMat(B.value .* (A.diag .\ B.value))
        function $(f)(A::ScalMat, B::PDiagMat)
            @check_argdims A.dim == size(B, 1)
            return PDiagMat(B.diag .* (A.value .\ B.diag))
        end
    end
end
