# Structured based on MLBase.jl https://github.com/JuliaStats/MLBase.jl

abstract type CrossValGenerator end

# K-fold

struct Kfold <: CrossValGenerator
    permseq::Vector{Int}
    k::Int
    coeff::Float64

    function Kfold(n::Int, k::Int)
        2 <= k <= n || error("The value of k must be in [2, length(a)].")
        new(randperm(n), k, n / k)
    end
end

length(c::Kfold) = c.k

struct KfoldState
    i::Int      # the i-th of the subset
    s::Int      # starting index
    e::Int      # ending index
end

# A version check allows to maintain compatibility with earlier versions
function Base.iterate(c::Kfold, state::KfoldState=KfoldState(1, 1, round.(Integer, c.coeff)))
    i, s, e = state.i, state.s, state.e
    (i > c.k) && return nothing
    i += 1
    sd = setdiff(1:length(c.permseq), c.permseq[s:e])
    kst = KfoldState(i, e + 1, round.(Integer, c.coeff * i))
    return sd, kst
end

# Stratified K-fold

struct StratifiedKfold <: CrossValGenerator
    n::Int                         #Total number of observations
    permseqs::Vector{Vector{Int}}  #Vectors of vectors of indexes for each stratum
    k::Int                         #Number of splits
    coeffs::Vector{Float64}        #About how many observations per strata are in a val set
    function StratifiedKfold(strata, k)
        2 <= k <= length(strata) || error("The value of k must be in [2, length(strata)].")
        strata_labels, permseqs = unique_inverse(strata)
        map(shuffle!, permseqs)
        coeffs = Float64[]
        for (stratum, permseq) in zip(strata_labels, permseqs)
            k <= length(permseq) || error("k is greater than the length of stratum $stratum")
            push!(coeffs, length(permseq) / k)
        end
        new(length(strata), permseqs, k, coeffs)
    end
end

length(c::StratifiedKfold) = c.k

function Base.iterate(c::StratifiedKfold, s::Int=1)
    (s > c.k) && return nothing
    r = Int[]
    for (permseq, coeff) in zip(c.permseqs, c.coeffs)
        a, b = round.(Integer, [s-1, s] .* coeff)
        append!(r, view(permseq, a+1:b))
    end
    return setdiff(1:c.n, r), s+1
end

function unique_inverse(A::AbstractArray)
    out = Array{eltype(A)}(undef, 0)
    out_idx = Array{Vector{Int}}(undef, 0)
    seen = Dict{eltype(A), Int}()
    for (idx, x) in enumerate(A)
        if !in(x, keys(seen))
            seen[x] = length(seen) + 1
            push!(out, x)
            push!(out_idx, Int[])
        end
        push!(out_idx[seen[x]], idx)
    end
    out, out_idx
end

function cross_validate(estfun::Function, evalfun::Function, n::Int, gen)
    best_model = nothing
    best_score = NaN
    best_inds = Int[]
    first = true

    scores = Float64[]
    for (i, train_inds) in enumerate(gen)
        #println(i, train_inds)
        test_inds = setdiff(1:n, train_inds)
        model = estfun(train_inds)
        score = evalfun(model, test_inds)
        push!(scores, score)
    end
    return scores
end

function cross_validate(u::COXUpdate, X::Matrix, δ::Vector, t::Vector, penalties::Vector{P}, k::Int; T=Float64, A=Array, mapper=Base.identity) where P <: Penalty 
    gen = StratifiedKfold(δ, k)
    n = size(X, 1)
    scores = Array{Float64}(undef, length(penalties), k)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        X_train = mapper(X[train_inds, :])
        X_test = mapper(X[test_inds, :])
        δ_train = δ[train_inds]
        δ_test = δ[test_inds]
        t_train = t[train_inds]
        t_test = t[test_inds]
        p = penalties[1]
        V = ProxCox.COXVariables{T, A}(adapt(A{T}, X_train), adapt(A{T}, δ_train), adapt(A{T}, t_train), p; eval_obj=true)
        for (i, p) in enumerate(penalties)
            V.penalty = p
            V.obj_prev = -Inf
            @time fit!(u, V)
            scores[i, j] = cindex(t_test, δ_test, X_test, adapt(Array{T}, V.β))
        end
    end
    scores
end