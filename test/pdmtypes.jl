# test pd matrix types

using PDMats

C1 = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
va = [1.5, 2.5, 2.0]

for (C, Cmat, ceq) in Any[
                          (PDMat(C1), C1, true),
                          (PDiagMat(va), diagm(va), true),
                          (ScalMat(3, 2.0), 2.0 * eye(3, 3), true),
                          (PDSparseMat(sparse(C1)), C1, true)
    ]

    test_pdmat(C, Cmat; cmat_eq=ceq, verbose=1)
end
