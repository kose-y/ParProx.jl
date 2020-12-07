using .CUDA
using Adapt

function COXVariables{T}(X::CuMatrix, X_unpen::CuMatrix, δ::CuVector, t::CuVector, lambda::T2,
    groups::Vector{Vector{Int}};
    σ=nothing, eval_obj=true) where {T <: Real, T2 <: Real}

    mapper, grpmat, grpidx = mapper_mat_idx(groups, size(X, 2); sparsemapper=(x)->
        CUSPARSE.CuSparseMatrixCSC{T}(convert(CuArray{Cint}, x.colptr), convert(CuArray{Cint}, x.rowval),
            adapt(CuArray{T}, x.nzval), size(x)))#, convert(Cint,length(x.nzval))))

    X_map = mapper(X, X_unpen)

    penalty = GroupNormL2{T, CuArray}(lambda, grpidx)

    COXVariables{T,CuArray}(X_map, δ, t, penalty; eval_obj=eval_obj)
end

function π_δ_kernel!(out, w, W, δ, breslow)
    # fill `out` with zeros beforehand.
    idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for i = idx_x:stride_x:length(out)
        for j = 1:length(out)
            @inbounds if breslow[i] <= breslow[j]
                out[i] += δ[j] * w[i] / W[j]
            end
        end
    end
end

function π_δ!(out::CuArray, w::CuArray, W::CuArray, δ::CuArray, breslow)
    numblocks = ceil(Int, length(w)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks π_δ_kernel!(out, w, W, δ, breslow)
    end
end
