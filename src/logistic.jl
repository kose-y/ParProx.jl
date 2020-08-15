using LinearAlgebra, LinearMaps
using Random, Adapt

"""
    SoftmaxUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10)

Stopping and evaluation rule for Logistic regression
"""
mutable struct LogisticUpdate
    maxiter::Int
    step::Int
    tol::Real
    verbose::Bool
    function LogisticUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10, verbose=true)
        maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        new(maxiter, step, tol, verbose)
    end
end

"""
    LogisticVariables{T, AT}(X::LinearMap, y::AbstractMatrix, penalty::Penalty; σ::Real=1/(2*power(X)^2), eval_obj::Bool=false)
    ) where {T <: Real, AT <: AbstractArray}

Setup variables for logistic regression given the data and penalty configuration.

# Arguments

- `X::MapOrMatrix`: Data matrix
- `y::AbstractVector`: class label
- `penalty::Penalty`: A penalty object.
- `σ::Real=4/(power(X; ArrayType=AT)^2)`: Step size.
- `eval_obj::Bool=false`: whether to evaluate the objective function
"""
mutable struct LogisticVariables{T,A}
    m::Int # rows
    n::Int # cols
    X::MapOrMatrix
    y::AbstractVector
    penalty::Penalty
    β::A
    β_prev::A
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence. 
    grad::A
    probs::A # space for c probabilities
    eval_obj::Bool
    obj_prev::Real
    function LogisticVariables{T,AT}(X::MapOrMatrix{T}, y::AbstractVector{<:Integer}, penalty::Penalty; 
                                σ::Real=4/(power(X; ArrayType=AT)^2), eval_obj::Bool=false
                               ) where {T <: Real, AT <: AbstractArray}
        m, n = size(X)

        β = AT{T}(undef, n)
        β_prev = AT{T}(undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))
        
        grad  = AT{T}(undef, n)
        probs = AT{T}(undef, m)

        new{T,AT}(m, n, LinearMap(X), y, penalty, β, β_prev, σ, grad, probs, eval_obj, -Inf)
    end
end

"""
    LogisticVariables{<:Real}(X::AbstractMatrix, X_unpen::AbstractMatrix, y::AbstractVector, lambda::Real, 
        groups::Vector{Vector{Int}}; eval_obj::Bool=false))

Setup variables for Logistic regression given the data and an overlapping group lasso penalty.

# Arguments

- `X::AbstractMatrix`: Penalized variables
- `X_unpen::AbstractMatrix`: Unpenalized variables
- `y::AbstractVector`: class label
- `λ::Real`: size of penalty
- `groups::Vector{Vector{Int}}`: each element denotes member variables of each group. A variable may appear in multiple groups.
- `eval_obj::Bool`: whether to evaluate the objective function
"""
function LogisticVariables{T}(X::Matrix, X_unpen::Matrix, y::AbstractVector, lambda::T2,
    groups::Vector{Vector{Int}};
    σ=nothing, eval_obj=true) where {T <: Real, T2 <: Real}

    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2))

    X_map = mapper(X, X_unpen)

    penalty = GroupNormL2(lambda, grpidx)

    LogisticVariables{T,Array}(X_map, y, penalty; eval_obj=eval_obj)
end

"""
    reset!(v::SoftmaxVariables)

Reset the coefficients to zero
"""
function reset!(v::LogisticVariables{T,A}) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end

function logistic!(out::AbstractArray{T}, z::AbstractArray{T}) where T
    out .= one(T) ./ (one(T) .+ exp.(-z))
end

function logistic(z::AbstractArray{T}) where T
    logistic!(similar(z), z)
end

function logistic_grad!(out, β, X, y, probs)
    mul!(probs, X, β)
    logistic!(probs, probs)
    probs .= (y .- probs)
    mul!(out, transpose(X), probs)
    out ./= size(X, 1)
end

"""
    grad!(v::MultinomialLogisticVariables)

Compute the gradient of multinomial logistic  likelihood based on the current status of v.
"""
grad!(v::LogisticVariables{T,A}) where {T,A} = logistic_grad!(v.grad, v.β, v.X, v.y, v.probs)

"""
    get_objective!(v::MultinomialLogisticVariables)

Computes the objective function
"""
function get_objective!(u::LogisticUpdate, v::LogisticVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad used as dummy
    nnz = sum(v.grad)

    if v.eval_obj
        mul!(v.probs, v.X, v.β)
        logistic!(v.probs, v.probs)
        obj = sum(v.y .* log.(v.probs) .+ (one(T) .- v.y) .* log.(one(T) .- v.probs)) / size(v.X, 1) .- value(v.penalty, v.β)
        reldiff = (abs(obj - v.obj_prev))/(abs(obj) + one(T))
        converged =  reldiff < u.tol
        v.obj_prev = obj
        return converged, (obj, reldiff, nnz)
    else
        v.grad .= abs.(v.β .- v.β_prev)
        return false, (maximum(v.grad), nnz)
    end
end

"""
    one_iter!(v::MultinomialLogisticVariables)

Update one iteration of proximal gradient of the Cox regression
"""
function one_iter!(v::LogisticVariables)
    copyto!(v.β_prev, v.β)
    grad!(v)
    prox!(v.β, v.penalty, v.β .+ v.σ .* v.grad)
end


"""
    fit!(u::SoftmaxUpdate, v::SoftmaxVariables)

Run full penalized regression
"""
function fit!(u::LogisticUpdate, v::LogisticVariables)
    loop!(u, one_iter!, get_objective!, v)
end

"""
    accuracy(y, X, β)

Compute accuracy
"""
function accuracy(y, X, β)
    y = adapt(Array{eltype(y)}, y)
    Xβ = adapt(Array{eltype(β)}, X * β)
    prediction = logistic(Xβ) .>= 0.5

    numerator = count(prediction .== y)
    denominator = length(y)

    numerator / denominator
end
