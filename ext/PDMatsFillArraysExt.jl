module PDMatsFillArraysExt

using PDMats: PDMats, LinearAlgebra
using FillArrays: FillArrays

function PDMats.AbstractPDMat(a::LinearAlgebra.Diagonal{T,<:FillArrays.AbstractFill{T,1}}) where {T<:Real}
    dim = size(a, 1)
    return PDMats.ScalMat(dim, FillArrays.getindex_value(a.diag))
end

end # module
