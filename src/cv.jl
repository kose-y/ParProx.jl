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

"""
    cross_validate(u::COXUpdate, X::Matrix, δ::Vector, t::Vector, penalties::Vector{P}, k::Int; T=Float64)

Perform `k`-fold cross validation for penalized Cox regression.

# Arguments

- `u::COXUpdate`: convergence setup.
- `X::AbstractMatrix`: Variables, penalized or unpenalized. Penalization is defined by the penalties.
- `δ::AbstractVector`: event indicator
- `t::AbstractVector`: observed time to event or censoring
- `penalties::Vector{P}`: a vector of penalties to be compared in CV
- `k`: fold.
- `T`: A type of AbstractFloat.
"""
function cross_validate(u::COXUpdate, X::Matrix, δ::Vector, t::Vector, penalties::Vector{P}, k::Int;
    T=Float64, A=Array, mapper=Base.identity, eval_obj=true) where P <: Penalty
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
        V = ParProx.COXVariables{T, A}(adapt(A{T}, X_train), adapt(A{T}, δ_train), adapt(A{T}, t_train), p; eval_obj=eval_obj)
        for (i, p) in enumerate(penalties)
            V.penalty = p
            V.obj_prev = -Inf
            @time fit!(u, V)
            scores[i, j] = cindex(t_test, δ_test, X_test, adapt(Array{T}, V.β))
        end
    end
    scores
end

"""
    cross_validate(u::COXUpdate, X::Matrix, X_unpen::Matrix, δ::Vector, t::Vector, groups::Vector{Vector{Int}}, lambdas::Vector{<:Real}, k::Int; T=Float64)

Perform `k`-fold cross validation for Cox regression with overlapping group lasso penalties.

# Arguments

- `u::COXUpdate`: convergence setup.
- `X::AbstractMatrix`: Penalized variables.
- `X_unpen::AbstractMatrix` Unpenalized variables.
- `δ::AbstractVector`: event indicator
- `t::AbstractVector`: observed time to event or censoring
- `groups::Vector{Vector{Int}}`: each element denotes member variables of each group. A variable may appear in multiple groups.
- `lambdas`: vector of λs to perform CV.
- `k`: fold.
- `T`: A type of AbstractFloat.
"""
function cross_validate(u::COXUpdate, X::AbstractMatrix, X_unpen::AbstractVecOrMat, δ::AbstractVector, t::AbstractVector,
    groups::Vector{Vector{Int}}, lambdas::AbstractVector{<:Real}, k::Integer;
    T=Float64, eval_obj=true)
    gen = StratifiedKfold(δ, k)
    n = size(X, 1)
    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2))
    scores = Array{Float64}(undef, length(lambdas), k)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        X_train = mapper(X[train_inds, :], X_unpen[train_inds, :])
        X_test = mapper(X[test_inds, :], X_unpen[test_inds, :])
        δ_train = δ[train_inds]
        δ_test = δ[test_inds]
        t_train = t[train_inds]
        t_test = t[test_inds]
        p = GroupNormL2(lambdas[1], grpidx)
        V = ParProx.COXVariables{T, Array}(X_train, δ_train, t_train, p; eval_obj=eval_obj)
        for (i, l) in enumerate(lambdas)
            p = GroupNormL2(l, grpidx)
            V.penalty = p
            V.obj_prev = -Inf
            @time fit!(u, V)
            scores[i, j] = cindex(t_test, δ_test, X_test, V.β)
        end
    end
    scores
end

"""
    cross_validate(u::LogisticUpdate, X::AbstractMatrix, X_unpen::AbstractMatrix, y::AbstractVector, groups::Vector{Vector{Int}}, lambdas::Vector{<:Real}, k::Int; T=Float64)

Perform `k`-fold cross validation for penalized logistic regression.

# Arguments

- `u::LogisticUpdate`: convergence setup.
- `X::AbstractMatrix`: Penalized variables.
- `X_unpen::AbstractMatrix` Unpenalized variables.
- `y::AbstractVector`: 0/1 class indicator
- `groups::Vector{Vector{Int}}`: each element denotes member variables of each group. A variable may appear in multiple groups.
- `lambdas`: vector of λs to perform CV.
- `k`: fold.
- `T`: A type of AbstractFloat.
"""
function cross_validate(u::LogisticUpdate, X::Matrix, y::Vector, penalties::Vector{P}, k::Int;
    T=Float64, A=Array, criteria=accuracy, mapper=Base.identity, eval_obj=true) where P <: Penalty
    gen = StratifiedKfold(y, k)
    n = size(X, 1)
    scores = Array{Float64}(undef, length(penalties), k)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        X_train = mapper(X[train_inds, :])
        X_test = mapper(X[test_inds, :])
        y_train = y[train_inds]
        y_test = y[test_inds]
        p = penalties[1]
        V = ParProx.LogisticVariables{T, Array}(X_train, y_train, p; eval_obj=eval_obj)
        for (i, p) in enumerate(penalties)
            V.penalty = p
            V.obj_prev = -Inf
            if u.verbose
                @time fit!(u, V)
            else
                fit!(u, V)
            end
            scores[i, j] = criteria(y_test, X_test, adapt(Array{T}, V.β))
        end
    end
    scores
end


"""
    cross_validate(u::LogisticUpdate, X::AbstractMatrix, X_unpen::AbstractMatrix, y::AbstractVector, groups::Vector{Vector{Int}}, lambdas::Vector{<:Real}, k::Int; T=Float64)

Perform `k`-fold cross validation for logistic regression with overlapping group lasso penalties.

# Arguments

- `u::LogisticUpdate`: convergence setup.
- `X::AbstractMatrix`: Penalized variables.
- `X_unpen::AbstractMatrix` Unpenalized variables.
- `y::AbstractVector`: 0/1 class indicator
- `groups::Vector{Vector{Int}}`: each element denotes member variables of each group. A variable may appear in multiple groups.
- `lambdas`: vector of λs to perform CV.
- `k`: fold.
- `T`: A type of AbstractFloat.
"""
function cross_validate(u::LogisticUpdate, X::AbstractMatrix, X_unpen::AbstractVecOrMat, y::AbstractVector, groups::Vector{Vector{Int}}, lambdas::Vector{<:Real}, k::Int;
    T=Float64, criteria=accuracy, eval_obj=true)
    gen = StratifiedKfold(y, k)
    n = size(X, 1)
    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2))
    scores = Array{Float64}(undef, length(lambdas), k)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        X_train = mapper(X[train_inds, :], X_unpen[train_inds, :])
        X_test = mapper(X[test_inds, :], X_unpen[test_inds, :])
        y_train = y[train_inds]
        y_test = y[test_inds]

        p = GroupNormL2(lambdas[1], grpidx)
        V = ParProx.LogisticVariables{T, Array}(X_train, y_train, p; eval_obj=eval_obj)
        for (i, l) in enumerate(lambdas)
            p = GroupNormL2(l, grpidx)
            V.penalty = p
            V.obj_prev = -Inf
            if u.verbose
                @time fit!(u, V)
            else
                fit!(u, V)
            end
            scores[i, j] = criteria(y_test, X_test, V.β)
        end
    end
    scores
end
