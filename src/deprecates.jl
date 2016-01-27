# deprecated functions

import Base: depwarn

function add!(a::Matrix, b::AbstractPDMat)
    depwarn("add! is deprecated in favor of pdadd!", :add!)
    pdadd!(a, b)
end

function add_scal!(a::Matrix, b::AbstractPDMat, c::Real)
    depwarn("add! is deprecated in favor of pdadd!", :add_scal!)
    pdadd!(a, b, c)
end

function add_scal(a::Matrix, b::AbstractPDMat, c::Real)
    depwarn("add_scal is deprecated in favor of pdadd", :add_scal)
    pdadd(a, b, c)
end

