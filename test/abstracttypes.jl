using Test, PDMats

@testset "AbstractMatrix fallback functionality" begin
    C = Cmat = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
    
    test_pdmat(C, Cmat;
        verbose=2,             # the level to display intermediate steps
        cmat_eq=true,          # require Cmat and Matrix(C) to be exactly equal
        t_diag=false,          # whether to test diag method
        t_cholesky=false,      # whether to test cholesky method
        t_scale=false,         # whether to test scaling
        t_add=false,           # whether to test pdadd
        t_det=false,           # whether to test det method
        t_logdet=false,        # whether to test logdet method
        t_eig=false,           # whether to test eigmax and eigmin
        t_mul=false,           # whether to test multiplication
        t_div=false,           # whether to test division
        t_quad=true,           # whether to test quad & invquad
        t_triprod=false,       # whether to test X_A_Xt, Xt_A_X, X_invA_Xt, and Xt_invA_X
        t_whiten=true          # whether to test whiten and unwhiten
    )

end