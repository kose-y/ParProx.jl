using .CuArrays, .CUDAnative, Adapt
for Pen in (:GroupLassoL2, :IndGroupBallL2) 
    @eval begin
        function ($Pen){T, CuArray}(λ::T, idx::Vector{Ti}; gidx=false) where {T <: Real, Ti <: Integer}
            λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g = _get_grouplasso_args(λ, idx)
            grpmat = CuArrays.CUSPARSE.CuSparseMatrixCSR(grpmat)
            gidx = adapt(CuArray{Ti}, gidx)
            change_idxs = adapt(CuArray{Ti}, change_idxs)
            sizes = adapt(CuArray{Ti}, sizes)
            max_norms = adapt(CuArray{T}, max_norms)
            tmp_p = adapt(CuArray{T}, tmp_p)
            tmp_g = adapt(CuArray{T}, tmp_g)
            return ($Pen){T, CuArray}(λ, grpmat, gidx, change_idxs, sizes, p, ngrps, max_norms, tmp_p, tmp_g)
        end
    end
end

function gather_kernel!(out, vec, ind)
    idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for i = idx_x: stride_x:length(out)
        out[i]=vec[ind[i]]
    end
end

function gather!(out::CuArray, vec::CuArray, ind)
    numblocks = ceil(Int, length(out)/256)
    CuArrays.@sync begin
        @cuda threads=256 blocks=numblocks gather_kernel!(out, vec, ind)
    end
    out
end
