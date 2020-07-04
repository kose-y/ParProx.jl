module ProxCox

import Base: length

export Penalty, NormL1, GroupNormL2, value, prox!
export power, CoxUpdate, CoxVariables, reset!,  get_objective!, fit!

using LinearMaps

const MapOrMatrix{T} = Union{LinearMap{T},AbstractMatrix{T}}

include("utils.jl")
include("penalties.jl")
include("cox.jl")
include("logistic.jl")
include("softmax.jl")
include("cv.jl")
using Requires
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
            include("cuda/utils.jl")
            include("cuda/penalties.jl")
            include("cuda/cox.jl")
        end
    end
end

end # module
