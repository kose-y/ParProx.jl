using .CuArrays, .CUDAnative
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
    CuArrays.@sync begin
        @cuda threads=256 blocks=numblocks π_δ_kernel!(out, w, W, δ, breslow)
    end
end

function breslow_kernel!(out, cumsum_w, bind)
    idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for i = idx_x: stride_x:length(out)
        out[i]=cumsum_w[bind[i]]
    end
end

function get_breslow!(out::CuArray, cumsum_w::CuArray, bind)
    numblocks = ceil(Int, length(out)/256)
    CuArrays.@sync begin
        @cuda threads=256 blocks=numblocks breslow_kernel!(out, cumsum_w, bind)
    end
end
