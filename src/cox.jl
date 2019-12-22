using LinearAlgebra, LinearMaps
using Random

function power(x::LinearMap{T}; maxiter::Int=1000, eps::AbstractFloat=1e-6) where T <: AbstractFloat
    s_prev = -Inf
    n, p = size(x)
    v_cpu = Array{T}(undef, p)
    randn!(v_cpu)
    v = similar(x, T, p)
    copyto!(v, v_cpu)
    xv = similar(x, T, n)
    mul!(xv, x, v)
    s = 0.0
    for i = 1:maxiter
        mul!(v, transpose(x), xv) #v = transpose(x) * xv
        v ./= norm(v)
        mul!(xv, x, v)
        s = norm(xv)
        if abs((s_prev - s)/s) < eps
            break
        end
        s_prev = s
        if mod(i, 100) == 0
            println("power iteration: $i $s")
        end
    end
    s
end

mutable struct COXUpdate
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function COXUpdate(; maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        new(maxiter, step, verbose, tol)
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

mutable struct COXVariables{T,A}
    m::Int # rows
    n::Int # cols
    X::LinearMap
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
    function COXVariables(X::LinearMap, δ::AbstractVector, t::AbstractVector, penalty::Penalty;
            σ::Real=1/(2*power(X)^2), eval_obj::Bool=false)
        T = eltype(X)
        m, n = size(X)
        A = typeof(X).name.wrapper
        β = A{T}(undef, n)
        β_prev = A{T}(undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))
        
        δ = convert(A{T}, δ)
        
        breslow = convert(A{Int}, breslow_ind(convert(Array, t)))
        
        grad  = A{T}(undef, n)
        w = A{T}(undef, m)
        W = A{T}(undef, m)
        q = A{T}(undef, m) 
        
        new{T,A}(m, n, LinearMap(X), penalty, β, β_prev, δ, t, breslow, σ, grad, w, W, q, eval_obj)
    end
end

function reset!(v::COXVariables{T,A}) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end

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
    out
end

cox_grad!(v::COXVariables{T,A}) where {T,A} = cox_grad!(v.grad, v.w, v.W, v.t, v.q, v.X, v.β, v.δ, v.breslow)

function update!(v::COXVariables{T,A}) where {T,A}
    cox_grad!(v)
    prox!(v.β, v.penalty, v.β .+ v.σ .* v.grad)
    #v.β .= soft_threshold.(v.β .+ v.σ .* v.grad, v.λ)
end

function get_objective!(v::COXVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad used as dummy
    nnz = sum(v.grad)
    
    if v.eval_obj
        v.w .= exp.(mul!(v.w, v.X, v.β))
        cumsum!(v.q, v.w) # q used as dummy
        #v.W .= v.q[v.breslow]
        gather!(v.W, v.q, v.breslow)
        obj = dot(v.δ, mul!(v.q, v.X, v.β) .- log.(v.W)) .- value(v.penalty) #v.λ .* sum(abs.(v.β))
        return false, (obj, nnz)
    else
        v.grad .= abs.(v.β_prev .- v.β)
        return false, (maximum(v.grad), nnz)
    end
end

function cox_one_iter!(u::COXUpdate, v::COXVariables)
    copyto!(v.β_prev, v.β)
    update!(v)
end

function loop!(u, iterfun, evalfun, args...)
    converged = false
    t = 0
    while !converged && t < u.maxiter
        t += 1
        iterfun(u, args...)
        if t % u.step == 0
            converged, monitor = evalfun(u, args...)
            println("$(t)\t$(monitor)")
        end
    end
end
function cox!(u::COXUpdate, v::COXVariables)
    loop!(u, cox_one_iter!, get_objective!, v)
end
