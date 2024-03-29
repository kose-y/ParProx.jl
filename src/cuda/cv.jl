function cross_validate(u::COXUpdate, X::CuMatrix, X_unpen::CuMatrix, δ::CuVector, t::CuVector, groups::Vector{Vector{Int}}, lambdas::Vector{<:Real}, k::Int;
    T=Float64, eval_obj=true, σ=nothing)
    gen = StratifiedKfold(δ, k)
    n = size(X, 1)
    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2); sparsemapper=(x)->
        CUSPARSE.CuSparseMatrixCSC{T}(convert(CuArray{Cint}, x.colptr), convert(CuArray{Cint}, x.rowval),
        adapt(CuArray{T}, x.nzval), size(x)))#, convert(Cint,length(x.nzval))))
    scores = Array{Float64}(undef, length(lambdas), k)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        X_train = mapper(X[train_inds, :], X_unpen[train_inds, :])
        X_test = mapper(X[test_inds, :], X_unpen[test_inds, :])
        δ_train = δ[train_inds]
        δ_test = δ[test_inds]
        t_train = t[train_inds]
        t_test = t[test_inds]
        p = GroupNormL2{T, CuArray}(lambdas[1], grpidx)
        if σ !== nothing
            V = ParProx.COXVariables{T, CuArray}(X_train, δ_train, t_train, p; eval_obj=eval_obj, σ=σ)
        else
            V = ParProx.COXVariables{T, CuArray}(X_train, δ_train, t_train, p; eval_obj=eval_obj)
        end
        for (i, l) in enumerate(lambdas)
            p = GroupNormL2{T, CuArray}(l, grpidx)
            V.penalty = p
            V.obj_prev = -Inf
            if u.verbose
                @time fit!(u, V)
            else
                fit!(u, V)
            end
            scores[i, j] = cindex(t_test, δ_test, X_test, V.β)
        end
    end
    scores
end

function cross_validate(u::LogisticUpdate, X::CuMatrix, X_unpen::CuMatrix, y::CuVector, groups::Vector{Vector{Int}}, lambdas::Vector{<:Real}, k::Int;
    T=Float64, criteria=accuracy, eval_obj=true)
    gen = StratifiedKfold(y, k)
    n = size(X, 1)
    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2); sparsemapper=(x)->
        CUSPARSE.CuSparseMatrixCSC{T}(convert(CuArray{Cint}, x.colptr), convert(CuArray{Cint}, x.rowval),
        adapt(CuArray{T}, x.nzval), size(x)))
    scores = Array{Float64}(undef, length(lambdas), k)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        X_train = mapper(X[train_inds, :], X_unpen[train_inds, :])
        X_test = mapper(X[test_inds, :], X_unpen[test_inds, :])
        y_train = y[train_inds]
        y_test = y[test_inds]
        p = GroupNormL2{T, CuArray}(lambdas[1], grpidx)
        V = ParProx.LogisticVariables{T, CuArray}(X_train, y_train, p; eval_obj=eval_obj)
        for (i, l) in enumerate(lambdas)
            p = GroupNormL2{T, CuArray}(l, grpidx)
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
