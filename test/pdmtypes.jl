# test pd matrix types
using PDMats
using Base.Test

call_test_pdmat(p::AbstractPDMat,m::Matrix) = test_pdmat(p,m,cmat_eq=true,verbose=1)

for T in [Float64,Float32]
  #test that all external constructors are accessible
  m = eye(T,2) ; @test PDMat(m,cholfact(m)).mat == PDMat(Symmetric(m)).mat == PDMat(m).mat == PDMat(cholfact(m)).mat
  d = ones(T,2) ; @test PDiagMat(d,d).inv_diag == PDiagMat(d).inv_diag
  x = one(T) ; @test ScalMat(2,x,x).inv_value == ScalMat(2,x).inv_value
  #for Julia v0.3, do not test sparse matrices with Float32 (see issue #14076)
  (T == Float64 || VERSION >= v"0.4.2") && (s = speye(T,2,2) ; @test PDSparseMat(s,cholfact(s)).mat == PDSparseMat(s).mat == PDSparseMat(cholfact(s)).mat)

  #test the functionality
  M = convert(Array{T,2},[4. -2. -1.; -2. 5. -1.; -1. -1. 6.])
  V = convert(Array{T,1},[1.5, 2.5, 2.0])
  X = convert(T,2.0)

  call_test_pdmat(PDMat(M),M) #tests of PDMat
  call_test_pdmat(PDiagMat(V),diagm(V)) #tests of PDiagMat
  call_test_pdmat(ScalMat(3,x),x*eye(T,3)) #tests of ScalMat
  #for Julia v0.3, do not test sparse matrices with Float32 (see issue #14076)
  (T == Float64 || VERSION >= v"0.4.2") && call_test_pdmat(PDSparseMat(sparse(M)),M)
end
