using .CUDA
function LogisticVariables{T}(X::CuMatrix, X_unpen::CuMatrix, y::CuVector, lambda::T2,
    groups::Array{Vector{Int}};
    σ=nothing, eval_obj=true) where {T <: Real, T2 <: Real}

    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2); sparsemapper=(x)->
        CUSPARSE.CuSparseMatrixCSC{T}(convert(CuArray{Cint}, x.colptr), convert(CuArray{Cint}, x.rowval),
            adapt(CuArray{T}, x.nzval), size(x)))

    X_map = mapper(X, X_unpen)

    penalty = GroupNormL2{T, CuArray}(lambda, grpidx)

    LogisticVariables{T,CuArray}(X_map, y, penalty; eval_obj=eval_obj)
end
