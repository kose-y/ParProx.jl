using LinearAlgebra, LinearMaps
using Random, Adapt
import SparseArrays: SparseMatrixCSC


"""
    COXUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10)

Stopping and evaluation rule for Cox regression
"""
mutable struct COXUpdate <: OptimConfig
    maxiter::Int
    step::Int
    tol::Real
    verbose::Bool
    function COXUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10, verbose::Bool=false)
        maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        new(maxiter, step, tol, verbose)
    end
end

"""
    breslow_ind(x)

returns indexes of result of cumsum corresponding to "W". `x` is assumed to be nonincreasing.
"""
function breslow_ind(x::AbstractVector)
    uniq = unique(x)
    lastinds = findlast.(isequal.(uniq), [x])
    invinds = findfirst.(isequal.(x), [uniq])
    lastinds[invinds]
end

"""
    COXVariables{T, A}(X::MapOrMatrix, δ::AbstractVector, t::AbstractVector, penalty::Penalty; σ::Real=1/(2*power(X)^2), eval_obj::Bool=false)
    ) where {T <: Real, AT <: AbstractArray}

Setup variables for Cox regression given the data and penalty configuration.

# Arguments

- `X::MapOrMatrix`: Data matrix
- `δ::AbstractVector`: Event indicator
- `t::AbstractVector`: Observed time to event or censoring
- `penalty::Penalty`: A penalty object.
- `σ::Real=1/(2*power(X; ArrayType=AT)^2)`: Step size.
- `eval_obj::Bool=false`: whether to evaluate the objective function
"""
mutable struct COXVariables{T,A}
    m::Int # rows
    n::Int # cols
    X::MapOrMatrix
    penalty::Penalty
    β::A
    β_prev::A
    δ::A # indicator for right censoring (0 if censored)
    t::A # timestamps, must be in nonincreasing order
    breslow::A
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence.
    grad::A
    w::A
    W::A
    q::A # (1 - π)δ
    eval_obj::Bool
    obj_prev::Real
    function COXVariables{T,AT}(X::MapOrMatrix, δ::AbstractVector,
                                t::AbstractVector, penalty::Penalty;
                                σ::Real=1/(2*power(X; ArrayType=AT)^2), eval_obj::Bool=false
                               ) where {T <: Real, AT <: AbstractArray}
        m, n = size(X)
        β = AT{T}(undef, n)
        β_prev = AT{T}(undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))

        δ = convert(AT{T}, δ)

        breslow = convert(AT{Int}, breslow_ind(convert(Array, t)))

        grad  = AT{T}(undef, n)
        w = AT{T}(undef, m)
        W = AT{T}(undef, m)
        q = AT{T}(undef, m)

        new{T,AT}(m, n, LinearMap(X), penalty, β, β_prev, δ, t, breslow, σ, grad, w, W, q, eval_obj, -Inf)
    end
end

"""
    COXVariables{<:Real}(X::AbstractMatrix, X_unpen::AbstractMatrix, δ::AbstractVector, t::AbstractVector, lambda::Real,
        groups::Vector{Vector{Int}}; eval_obj::Bool=false))

Setup variables for Cox regression given the data and an overlapping group lasso penalty.

# Arguments

- `X::AbstractMatrix`: Penalized variables
- `X_unpen::AbstractMatrix`: Unpenalized variables
- `δ::AbstractVector`: Event indicator
- `t::AbstractVector`: Observed time to event or censoring
- `λ::Real`: size of penalty
- `groups::Vector{Vector{Int}}`: each element denotes member variables of each group. A variable may appear in multiple groups.
- `eval_obj::Bool`: whether to evaluate the objective function
"""
function COXVariables{T}(X::AbstractMatrix, X_unpen::AbstractMatrix, δ::AbstractVector, t::AbstractVector, lambda::T2,
    groups::Vector{Vector{Int}}; eval_obj=true) where {T <: Real, T2 <: Real}

    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2))

    X_map = mapper(X, X_unpen)

    penalty = GroupNormL2(lambda, grpidx)

    COXVariables{T,Array}(X_map, δ, t, penalty; eval_obj=eval_obj)
end

"""
    reset!(v::COXVariables)

Reset the coefficients to zero
"""
function reset!(v::COXVariables{T,A}) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end

"""
    π_δ!(out, w, W, δ, breslow)

compute out[i] = δ[j] * w[i]/ W[j] * I(breslow[i] <= breslow[j]).
"""
function π_δ!(out, w, W, δ, breslow)
    # fill `out` with zeros beforehand.
    m = length(δ)
    Threads.@threads for i in 1:m
        for j in 1:m
            @inbounds if breslow[i] <= breslow[j]
                out[i] +=  δ[j] * w[i]/ W[j]
            end
        end
    end
    out
end

function cox_grad!(out, w, W, t, q, X, β, δ, bind)
    T = eltype(β)
    m, n = size(X)
    mul!(w, X, β)
    w .= exp.(w)
    cumsum!(q, w) # q is used as a dummy variable
    gather!(W, q, bind)
    fill!(q, zero(eltype(q)))
    π_δ!(q, w, W, δ, bind)
    q .= δ .- q
    mul!(out, transpose(X), q)
    out = out ./ n
    out
end

"""
    grad!(v::COXVariables)

Compute the gradient of Cox partial likelihood based on the current status of v.
"""
grad!(v::COXVariables{T,A}) where {T,A} = cox_grad!(v.grad, v.w, v.W, v.t, v.q, v.X, v.β, v.δ, v.breslow)

"""
    get_objective!(v::COXVariables)

Computes the objective function
"""
function get_objective!(u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad used as dummy
    nnz = sum(v.grad)

    if v.eval_obj
        v.w .= exp.(mul!(v.w, v.X, v.β))
        cumsum!(v.q, v.w) # q used as dummy
        #v.W .= v.q[v.breslow]
        gather!(v.W, v.q, v.breslow)
        obj = (dot(v.δ, mul!(v.q, v.X, v.β) .- log.(v.W))) ./ size(v.X, 1) .- value(v.penalty, v.β) #v.λ .* sum(abs.(v.β))
        reldiff = (abs(obj - v.obj_prev))/(abs(obj) + one(T))
        converged =  reldiff < u.tol
        v.obj_prev = obj
        return converged, (obj, reldiff, nnz)
    else
        v.grad .= (v.β_prev .- v.β)
        relchange = norm(v.grad) / (norm(v.β) + 1e-20)
        converged = relchange < u.tol
        return converged, (relchange, norm(v.β),  nnz)
    end
end

"""
    cox_one_iter!(v::COXVariables)

Update one iteration of proximal gradient of the Cox regression
"""
function one_iter!(v::COXVariables)
    copyto!(v.β_prev, v.β)
    grad!(v)
    prox!(v.β, v.penalty, v.β .+ v.σ .* v.grad)
    #v.β .= soft_threshold.(v.β .+ v.σ .* v.grad, v.λ)
end


"""
    fit!(u::COXUpdate, v::COXVariables)

Run full Cox regression
"""
function fit!(u::COXUpdate, v::COXVariables)
    loop!(u, one_iter!, get_objective!, v)
end

"""
    cindex(v::COXVariables)
    cindex(t, δ, X, β)

Compute c-index
"""
function cindex(t, δ, X, β)
    t = adapt(Array{eltype(t)}, t)
    δ = adapt(Array{eltype(δ)}, δ)
    Xβ = adapt(Array{eltype(β)}, X * β)
    numerator = 0
    denominator = 0

    for i in 1:length(t)
        for j in (i+1):length(t)
            t[i] == t[j] && continue
            δ[j] == 0 && continue
            denominator += 1
            if Xβ[i] < Xβ[j]
                numerator += 1
            end
        end
    end

    numerator / denominator
end

function cindex(v::COXVariables)
    cindex(v.t, v.δ, v.X, v.β)
end
