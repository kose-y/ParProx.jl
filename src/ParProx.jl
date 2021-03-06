module ParProx

import Base: length
import ROCAnalysis: roc, AUC
export Penalty, NormL1, GroupNormL2, value, prox!
export power, COXUpdate, COXVariables, reset!,  get_objective!, fit!
export cindex, accuracy, auc

using LinearMaps

const MapOrMatrix{T} = Union{LinearMap{T},AbstractMatrix{T}}

datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)

include("utils.jl")
include("penalties.jl")
include("cox.jl")
include("logistic.jl")
include("softmax.jl")
include("cv.jl")
using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cuda/utils.jl")
        include("cuda/penalties.jl")
        include("cuda/cox.jl")
        include("cuda/logistic.jl")
        include("cuda/cv.jl")
    end
end

end # module
