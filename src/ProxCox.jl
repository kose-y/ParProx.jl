module ProxCox

export Penalty, NormL1, GroupNormL2, value, prox!
export power, CoxUpdate, CoxVariables, reset!, cox_grad!, get_objective!, cox!

include("utils.jl")
include("penalties.jl")
include("cox.jl")

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
