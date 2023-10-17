using FillArrays

@testset "PDMatsFillArraysExt" begin
    for diag in (Ones(5), Fill(4.1, 8))
        a = @inferred(AbstractPDMat(Diagonal(diag)))
        @test a isa ScalMat
        @test a.dim == length(diag)
        @test a.value == first(diag)
    end
end
