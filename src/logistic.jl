using LinearAlgebra, LinearMaps
using Random, Adapt

"""
    SoftmaxUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10)

Stopping and evaluation rule for Cox regression
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
    SoftmaxVariables(X::LinearMap, y::AbstractMatrix, penalty::Penalty; σ::Real=1/(2*power(X)^2), eval_obj::Bool=false))

Setup variables for multinomial logistic regression given the data and penalty configuration.
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

        new{T,AT}(m, n, LinearMap(X), y, penalty, β, β_prev, σ, grad, probs, eval_obj)
    end
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
function get_objective!(v::LogisticVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad used as dummy
    nnz = sum(v.grad)

    if v.eval_obj
        mul!(v.probs, v.X, v.β)
        logistic!(v.probs, v.probs)
        obj = sum(v.y .* log.(v.probs) .+ (one(T) .- v.y) .* log.(one(T) .- v.probs)) / size(v.X, 1) .- value(v.penalty, v.β)
        return false, (obj, nnz)
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

