using LinearAlgebra, LinearMaps
using Random, Adapt

"""
    SoftmaxUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10)

Stopping and evaluation rule for Cox regression
"""
mutable struct SoftmaxUpdate
    maxiter::Int
    step::Int
    tol::Real
    function SoftmaxUpdate(; maxiter::Int=100, step::Int=10, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        new(maxiter, step, tol)
    end
end

@inline function to_onehot(y::AbstractVector{<:Integer})
    c = maximum(y)
    onehot = zeros(size(y, 1), c)
    for (i,v) in enumerate(y)
        onehot[i, v] = 1
    end
    return onehot
end

"""
    SoftmaxVariables(X::LinearMap, y::AbstractMatrix, penalty::Penalty; σ::Real=1/(2*power(X)^2), eval_obj::Bool=false))

Setup variables for multinomial logistic regression given the data and penalty configuration.
"""
mutable struct SoftmaxVariables{T,A}
    m::Int # rows
    n::Int # cols
    X::MapOrMatrix
    y::AbstractVector
    y_onehot::AbstractMatrix
    penalty::Penalty
    β::A
    β_prev::A
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence. 
    grad::A
    probs::A # space for c probabilities
    eval_obj::Bool
    function SoftmaxVariables{T,AT}(X::MapOrMatrix{T}, y::AbstractVector{<:Integer}, penalty::Penalty; 
                                σ::Real=1/(power(X; ArrayType=AT)), eval_obj::Bool=false
                               ) where {T <: Real, AT <: AbstractArray}
        m, n = size(X)
        y_onehot = to_onehot(y)
        y_onehot = adapt(AT{T}, y_onehot)
        c = size(y_onehot, 2)

        β = AT{T}(undef, n, c)
        β_prev = AT{T}(undef, n, c)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))
        
        grad  = AT{T}(undef, n, c)
        probs = AT{T}(undef, m, c)

        new{T,AT}(m, n, LinearMap(X), y, y_onehot, penalty, β, β_prev, σ, grad, probs, eval_obj)
    end
end

"""
    reset!(v::SoftmaxVariables)

Reset the coefficients to zero
"""
function reset!(v::SoftmaxVariables{T,A}) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end

function softmax!(out, z)
    out .= z .- maximum(z; dims=2)
    out .= exp.(out)
    out .= out ./ sum(out; dims=2)
end

function softmax(z::AbstractArray{T}) where T
    softmax!(similar(z), z)
end

function softmax_grad!(out, β, X, y_onehot, probs)
    mul!(probs, X, β)
    softmax!(probs, probs)
    probs .= (y_onehot .- probs)
    mul!(out, transpose(X), probs)
    out ./= size(X, 1)
end

"""
    grad!(v::MultinomialLogisticVariables)

Compute the gradient of multinomial logistic  likelihood based on the current status of v.
"""
grad!(v::SoftmaxVariables{T,A}) where {T,A} = softmax_grad!(v.grad, v.β, v.X, v.y_onehot, v.probs)

"""
    get_objective!(v::MultinomialLogisticVariables)

Computes the objective function
"""
function get_objective!(v::SoftmaxVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad used as dummy
    nnz = sum(v.grad)

    if v.eval_obj
        mul!(v.probs, v.X, v.β)
        softmax!(v.probs, v.probs)
        obj = sum(v.y_onehot .* log.(v.probs))/ size(v.X, 1) .- value(v.penalty, v.β)
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
function one_iter!(v::SoftmaxVariables)
    copyto!(v.β_prev, v.β)
    grad!(v)
    prox!(v.β, v.penalty, v.β .+ v.σ .* v.grad)
    # v.β .= soft_threshold.(v.β .+ v.σ .* v.grad, v.λ)
end


"""
    fit!(u::SoftmaxUpdate, v::SoftmaxVariables)

Run full penalized regression
"""
function fit!(u::SoftmaxUpdate, v::SoftmaxVariables)
    loop!(u, one_iter!, get_objective!, v)
end
