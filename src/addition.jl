
+ (a::PDMat,    b::Matrix{Float64}) = a.mat + b
+ (a::PDiagMat, b::Matrix{Float64}) = add_diag(b, a.diag)
+ (a::ScalMat,  b::Matrix{Float64}) = add_diag(b, a.value)

+ (a::Matrix{Float64}, b::AbstractPDMat) = b + a

function add!(a::Matrix{Float64}, b::PDMat)
    bm = b.mat
    @check_argdims size(a) == size(bm)
    for i = 1:length(a)
        @inbounds a[i] += bm[i]
    end
    a
end

add!(a::Matrix{Float64}, b::PDiagMat) = add_diag!(a, b.diag)
add!(a::Matrix{Float64}, b::ScalMat) = add_diag!(a, b.value)

add_scal!(a::Matrix{Float64}, b::PDMat, c::Float64) = axpy!(c, b.mat, a)
add_scal!(a::Matrix{Float64}, b::PDiagMat, c::Float64) = add_diag!(a, b.diag, c)
add_scal!(a::Matrix{Float64}, b::ScalMat, c::Float64) = add_diag!(a, b.value * c)

add_scal(a::Matrix{Float64}, b::AbstractPDMat, c::Float64) = add_scal!(copy(a), b, c)

# between pdmat and pdmat

+ (a::PDMat, b::AbstractPDMat) = PDMat(a.mat + full(b))
+ (a::PDiagMat, b::AbstractPDMat) = PDMat(add_diag!(full(b), a.diag))
+ (a::ScalMat, b::AbstractPDMat) = PDMat(add_diag!(full(b), a.value))

+ (a::PDMat, b::PDMat) = PDMat(a.mat + b.mat)
+ (a::PDMat, b::PDiagMat) = PDMat(add_diag(a.mat, b.diag))
+ (a::PDMat, b::ScalMat) = PDMat(add_diag(a.mat, b.value))

+ (a::PDiagMat, b::PDMat) = PDMat(add_diag(b.mat, a.diag))
+ (a::PDiagMat, b::PDiagMat) = PDiagMat(a.diag + b.diag)
+ (a::PDiagMat, b::ScalMat) = PDiagMat(a.diag .+ b.value)

+ (a::ScalMat, b::PDMat) = PDMat(add_diag(b.mat, a.value))
+ (a::ScalMat, b::PDiagMat) = PDiagMat(a.value .+ b.diag)
+ (a::ScalMat, b::ScalMat) = ScalMat(a.dim, a.value + b.value)

add_scal(a::PDMat, b::AbstractPDMat, c::Float64) = PDMat(a.mat + full(b * c))
add_scal(a::PDiagMat, b::AbstractPDMat, c::Float64) = PDMat(add_diag!(full(b * c), a.diag))
add_scal(a::ScalMat, b::AbstractPDMat, c::Float64) = PDMat(add_diag!(full(b * c), a.value))

add_scal(a::PDMat, b::PDMat, c::Float64) = PDMat(a.mat + b.mat * c)
add_scal(a::PDMat, b::PDiagMat, c::Float64) = PDMat(add_diag(a.mat, b.diag, c))
add_scal(a::PDMat, b::ScalMat, c::Float64) = PDMat(add_diag(a.mat, b.value * c))

add_scal(a::PDiagMat, b::PDMat, c::Float64) = PDMat(add_diag!(b.mat * c, a.diag))
add_scal(a::PDiagMat, b::PDiagMat, c::Float64) = PDiagMat(a.diag + b.diag * c)
add_scal(a::PDiagMat, b::ScalMat, c::Float64) = PDiagMat(a.diag .+ b.value * c)

add_scal(a::ScalMat, b::PDMat, c::Float64) = PDMat(add_diag!(b.mat * c, a.value))
add_scal(a::ScalMat, b::PDiagMat, c::Float64) = PDiagMat(a.value .+ b.diag * c)
add_scal(a::ScalMat, b::ScalMat, c::Float64) = ScalMat(a.dim, a.value + b.value * c)
