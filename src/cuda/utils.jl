using .CUDA
function gather_kernel!(out, vec, ind)
    idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for i = idx_x: stride_x:length(out)
        out[i]=vec[ind[i]]
    end
end

function gather!(out::CuArray, vec::CuArray, ind)
    numblocks = ceil(Int, length(out)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gather_kernel!(out, vec, ind)
    end
    out
end
