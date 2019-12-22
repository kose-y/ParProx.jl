using SparseArrays, LinearAlgebra
abstract type Penalty end

function soft_threshold(x::T, λ::T) ::T where T <: AbstractFloat
    @assert λ >= 0 "Argument λ must exceed 0"
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

struct NormL1{T<:Real, ArrayType<:AbstractArray} <: Penalty
    λ::T
end

function NormL1(λ::T; ArrayType=Array) where T<:Real
    return NormL1{T, ArrayType}(λ)
end

function prox!(y::AbstractArray{T}, f::NormL1{T,A}, x::AbstractArray{T}, γ::T=one(T)) where {T <: Real, A<:AbstractArray}
    y .= soft_threshold.(x, γ .* f.λ)
    y
end

function value(f::NormL1{T,A}, x::AbstractArray{T}) where {T <: Real, A <: AbstractArray}
    return sum(abs.(x))
end


function _get_grouplasso_args(λ::T, idx::Vector{Ti}) where {T <: Real, Ti <: Integer}
    @assert idx[1] == 1
    diff = idx[2:end] - idx[1:end-1]
    @assert all(0 .≤ diff .≤ 1)
    l = length(idx)
    ngrps = idx[end]
    gidx = idx
    change_idxs = vcat([1], (2:l)[diff .== 1], l+1)

    sizes = change_idxs[2:end].- change_idxs[1:end-1]
    grpmat = SparseMatrixCSC{T, Ti}(l, ngrps, change_idxs, collect(1:l), ones(T, l))
    p = size(grpmat, 1)
    max_norms = λ .* sqrt.(sizes)
    tmp_p = Vector{T}(undef, p)
    tmp_g = Vector{T}(undef, ngrps)
    return λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p,tmp_g
end

function gather!(out, vec, ind)
    out .= vec[ind]
end

for Pen in (:GroupNormL2, :IndGroupBallL2) 
    @eval begin
        struct ($Pen){T<:Real, ArrayType<:AbstractArray} <: Penalty
            λ::T
            grpmat::AbstractSparseMatrix
            gidx::AbstractVector{<:Integer}
            change_idxs::AbstractVector{<:Integer}
            sizes::AbstractVector{T}
            p::Integer
            ngrps::Integer
            max_norms::AbstractVector{T}
            tmp_p::AbstractVector{T}
            tmp_g::AbstractVector{T}
        end
        function ($Pen)(λ::T, idx::Vector{Ti}) where {T <: Real, Ti <: Integer}
            ArrayType=Array
            λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g = _get_grouplasso_args(λ, idx)
            return ($Pen){T,ArrayType}(λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g)
        end
    end
end

for (Pen1, Pen2) in [(:GroupNormL2, :IndGroupBallL2), (:IndGroupBallL2, :GroupNormL2)]
    @eval begin
        function ($Pen1)(dualpen::($Pen2){T, ArrayType}) where {T <: Real, ArrayType <: AbstractArray}
            return ($Pen1){T, ArrayType}(dualpen.λ, dualpen.grpmat, dualpen.gidx, dualpen.change_idxs, dualpen.sizes, dualpen.p, dualpen.ngrps, dualpen.max_norms, dualpen.tmp_p, dualpen.tmp_g)
        end
    end
end

function prox!(y::AbstractArray{T}, f::IndGroupBallL2{T,A}, x::AbstractArray{T}, γ::T=one(T)) where {T <: Real, A<:AbstractArray}
    y .= x .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), y)
    f.tmp_g .= sqrt.(f.tmp_g) # groupwise norms
    f.tmp_g .= f.max_norms ./ (max(f.max_norms, f.tmp_g))
    gather!(f.tmp_p, f.tmp_g, f.gidx)
    y .= x .* f.tmp_p
    y
end

function prox!(y::AbstractArray{T}, f::GroupNormL2{T,A}, x::AbstractArray{T}, γ::T=one(T)) where {T <: Real, A<:AbstractArray}
    y .= x .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), y)
    f.tmp_g .= sqrt.(f.tmp_g) # groupwise norms
    f.tmp_g .= γ * f.max_norms ./ (max(γ * f.max_norms, f.tmp_g))
    gather!(f.tmp_p, f.tmp_g, f.gidx)
    y .= x .- x .* f.tmp_p
    y
end

function value(f::IndGroupBallL2{T,A}, x::AbstractArray{T}) where {T <: Real, A <: AbstractArray}
    f.tmp_p .= x .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), f.tmp_p)
    f.tmp_g .= sqrt.(f.tmp_g)
    return sum(sqrt.(f.sizes) .* f.tmp_g) - f.λ > eps(T) * f.λ ? T(Inf) : zero(T)
end

function value(f::GroupNormL2{T,A}, x::AbstractArray{T}) where {T <: Real, A <: AbstractArray}
    f.tmp_p .= x .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), f.tmp_p)
    f.tmp_g .= sqrt.(f.tmp_g)
    return f.λ .* sum(sqrt.(f.sizes) .* f.tmp_g)
end