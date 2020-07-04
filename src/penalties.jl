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
    unpen::Int
end

function NormL1(λ::T; ArrayType=Array, unpen::Int=0) where T<:Real
    return NormL1{T, ArrayType}(λ, unpen)
end

"""
    prox!(y, f, x, γ; unpen)

update `y` with the proximity operator value `prox_{γf}(x)`, with last `unpen` variables unpenalized.
"""
function prox!(y::AbstractArray{T}, f::NormL1{T,A}, x::AbstractArray{T}, γ::T=one(T); unpen::Int=f.unpen) where {T <: Real, A<:AbstractArray}
    y[1:end-unpen] .= soft_threshold.(@view(x[1:end-unpen]), γ .* f.λ)
    y[end-unpen+1:end] .= @view(x[end-unpen+1:end])
    y
end

"""
    value(f, x; unpen)

value of f(x), with last `unpen` variables unpenalized.
"""
function value(f::NormL1{T,A}, x::AbstractArray{T}; unpen::Int=f.unpen) where {T <: Real, A <: AbstractArray}
    return f.λ * sum(abs.(@view(x[1:end-unpen])))
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

function _get_grouplasso_args_rowwise(λ::T, idx::Vector{Ti}, ncols::Int) where {T <: Real, Ti <: Integer}
    # grpmat stands, gidx stands, sizes multiplied by ncols, p multiplied by ncols (?), ngrp stands, max_norms multiplied by sqrt(ncols), 
    # tmp_p becomes a matrix, tmp_g stands. 
end

function _get_grouplasso_args_sepcols(λ::T, idx::Vector{Ti}, ncols::Int) where {T <: Real, Ti <: Integer}
    # grpmat repeated horizontally ncols times, offsetting needed for gidx, sizes repeated ncols times, p multiplied by ncols (?), 
    # ngrp multiplied by ncols, max_norms repeated, tmp_p becomes a matrix (or a longer vector), tmp_g becomes a matrix (or longer vector)
end

for Pen in (:GroupNormL2, :IndGroupBallL2) 
    @eval begin
        struct ($Pen){T<:Real, ArrayType<:AbstractArray} <: Penalty
            λ::T
            grpmat::LinearMap
            gidx::AbstractArray{<:Integer}
            change_idxs::AbstractArray{<:Integer}
            sizes::AbstractArray{T}
            p::Integer
            ngrps::Integer
            max_norms::AbstractVector{T}
            tmp_p::AbstractArray{T}
            tmp_g::AbstractArray{T}
            ncols::Int
            rowwise::Bool
        end
        function ($Pen)(λ::T, idx::Vector{Ti}) where {T <: Real, Ti <: Integer}
            ArrayType=Array
            λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g = _get_grouplasso_args(λ, idx)
            return ($Pen){T,ArrayType}(λ, LinearMap(grpmat), gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g, 1, false)
        end

        function ($Pen)(λ::T, idx::Vector{Ti}, ncols::Int, rowwise::Bool) where {T <: Real, Ti <: Integer}
            ArrayType=Array
            #λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g = _get_grouplasso_args(λ, idx)
            #some special setups depending on the value of rowwise
            if rowwise
                #TODO
            
            
            else
                #TODO
            
            
            end
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
    f.tmp_g .= f.max_norms ./ (max.(f.max_norms, f.tmp_g))
    gather!(f.tmp_p, f.tmp_g, f.gidx)
    y .= x .* f.tmp_p
    y
end

function prox!(y::AbstractArray{T}, f::GroupNormL2{T,A}, x::AbstractArray{T}, γ::T=one(T); unpen::Int=length(x) - f.p) where {T <: Real, A <: AbstractArray}
    y[1:end-unpen] .= @view(x[1:end-unpen]) .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), @view(y[1:end-unpen]))
    f.tmp_g .= sqrt.(f.tmp_g) # groupwise norms
    f.tmp_g .= (γ .* f.max_norms) ./ (max.(γ .* f.max_norms, f.tmp_g)) # multiplication factors
    gather!(f.tmp_p, f.tmp_g, f.gidx)
    y[1:end-unpen] .= (1 .- f.tmp_p) .* @view(x[1:end-unpen])
    y[end-unpen+1:end] .= x[end-unpen+1:end]
    y
end


#=
function prox!(y::AbstractMatrix{T}, f::GroupNormL2{T,A}, x::AbstractMatrix{T}, γ::T=one(T); unpen::Int=size(x, 1) - f.p, rowwise=false, 
               tmp_g::AbstractMatrix{T} = rowwise ? similar(x, f.ngrps, size(x, 2)) : similar(x, f.ngrps * size(x, 2))) where {T <: Real, A <: AbstractArray}
    y[1:end-unpen, :] .= @view(x[1:end-unpen, :]) .^ 2 

    if rowwise
        grpmat = kron(ones(1, ), f.grpmat)
    else
        max_norms = repeat(f.max_norms, size(x, 2))
        gidx = repeat(f.gidx, ) # TODO: offseting idxs
        grpmat = kron(ones(size(x, 2), 1), f.grpmat)
        mul!(tmp_g, transpose(grpmat), reshape(@view(y[1:end-unpen, :], :)))
        tmp_g .= sqrt.(tmp_g)
        tmp_g .= (γ .* max_norms) ./ (max.(γ .* max_norms, tmp_g))
        gather!()

    end

    y[1:end-unpen, :] .= (1 .- tmp_p) .* @view(x[1:end-unpen, :])

    y[end-unpen+1:end, :] .= x[end-unpen+1:end, :]
end
=#

function value(f::IndGroupBallL2{T,A}, x::AbstractVector{T}) where {T <: Real, A <: AbstractArray}
    f.tmp_p .= x .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), f.tmp_p)
    f.tmp_g .= sqrt.(f.tmp_g)
    return sum(sqrt.(f.sizes) .* f.tmp_g) - f.λ > eps(T) * f.λ ? T(Inf) : zero(T)
end

function value(f::GroupNormL2{T,A}, x::AbstractVector{T}; unpen::Int=length(x) - f.p) where {T <: Real, A <: AbstractArray}
    f.tmp_p .= @view(x[1:end-unpen]) .^ 2
    mul!(f.tmp_g, transpose(f.grpmat), f.tmp_p)
    f.tmp_g .= sqrt.(f.tmp_g)
    return f.λ .* sum(sqrt.(f.sizes) .* f.tmp_g)
end

#=
function value(f::GroupNormL2{T,A}, x::AbstractMatrix{T}; unpen::Int=size(x, 1) - f.p, rowwise=false, 
               tmp_g::AbstractMatrix{T}=similar(x, f.ngrps, size(x,2)), tmp_p::AbstractMatrix{T}=similar(x, size(x,1)-unpen, size(x, 2))) where {T <: Real, A <: AbstractArray}

    tmp_p .= @view(x[1:end-unpen, :]) .^ 2
    #TODO
    tmp_g .= sqrt.(tmp_g)
    #TODO
end
=#
