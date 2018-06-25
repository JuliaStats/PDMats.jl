# test pd matrix types
using LinearAlgebra, PDMats, SparseArrays, SuiteSparse
using Test

call_test_pdmat(p::AbstractPDMat,m::Matrix) = test_pdmat(p,m,cmat_eq=true,verbose=1)

for T in [Float64, Float32]
    #test that all external constructors are accessible
    m = Matrix{T}(I, 2, 2)
    @test PDMat(m, cholesky(m)).mat == PDMat(Symmetric(m)).mat == PDMat(m).mat == PDMat(cholesky(m)).mat
    d = ones(T,2)
    @test PDiagMat(d,d).inv_diag == PDiagMat(d).inv_diag
    x = one(T)
    @test ScalMat(2,x,x).inv_value == ScalMat(2,x).inv_value
    s = SparseMatrixCSC{T}(I, 2, 2)
    @test PDSparseMat(s, cholesky(s)).mat == PDSparseMat(s).mat == PDSparseMat(cholesky(s)).mat

    #test the functionality
    M = convert(Array{T,2}, [4. -2. -1.; -2. 5. -1.; -1. -1. 6.])
    V = convert(Array{T,1}, [1.5, 2.5, 2.0])
    X = convert(T,2.0)

    call_test_pdmat(PDMat(M), M) #tests of PDMat
    call_test_pdmat(PDiagMat(V), Matrix(Diagonal(V))) #tests of PDiagMat
    call_test_pdmat(ScalMat(3,x), x*Matrix{T}(I, 3, 3)) #tests of ScalMat
    call_test_pdmat(PDSparseMat(sparse(M)), M)
end

m = Matrix{Float32}(I, 2, 2)
@test convert(PDMat{Float64}, PDMat(m)).mat == PDMat(convert(Array{Float64}, m)).mat
@test convert(AbstractArray{Float64}, PDMat(m)).mat == PDMat(convert(Array{Float64}, m)).mat
m = ones(Float32,2)
@test convert(PDiagMat{Float64}, PDiagMat(m)).diag == PDiagMat(convert(Array{Float64}, m)).diag
@test convert(AbstractArray{Float64}, PDiagMat(m)).diag == PDiagMat(convert(Array{Float64}, m)).diag
x = one(Float32); d = 4
@test convert(ScalMat{Float64}, ScalMat(d, x)).value == ScalMat(d, convert(Float64, x)).value
@test convert(AbstractArray{Float64}, ScalMat(d, x)).value == ScalMat(d, convert(Float64, x)).value
s = SparseMatrixCSC{Float32}(I, 2, 2)
@test convert(PDSparseMat{Float64}, PDSparseMat(s)).mat == PDSparseMat(convert(SparseMatrixCSC{Float64}, s)).mat
