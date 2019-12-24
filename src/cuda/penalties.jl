using .CuArrays, .CUDAnative, Adapt
for Pen in (:GroupNormL2, :IndGroupBallL2) 
    @eval begin
        function ($Pen){T, CuArray}(λ::Real, idx::Vector{Ti}; gidx=false) where {T <: Real, Ti <: Integer}
            λ = convert(T, λ)
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
