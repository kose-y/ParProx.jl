"""
    gather!(out, vec, ind)

Simply returns out .= vec[ind]. Intended for extension on `CuArray`s.
"""
function gather!(out, vec, ind)
    out .= vec[ind]
end

"""
    power(x::LinearMap{T})

Power iteration to compute induced operator norm of the `Matrix` (or `LinearMap`) `x`.
"""
function power(x::LinearMap{T}; ArrayType=Array, maxiter::Int=1000, eps::AbstractFloat=1e-6) where T <: AbstractFloat
    s_prev = -Inf
    n, p = size(x)
    v_cpu = Array{T}(undef, p)
    randn!(v_cpu)
    v = adapt(ArrayType{T}, v_cpu)
    xv = similar(v, T, n)
    mul!(xv, x, v)
    s = 0.0
    for i = 1:maxiter
        mul!(v, transpose(x), xv) #v = transpose(x) * xv
        v ./= norm(v)
        mul!(xv, x, v)
        s = norm(xv)
        if abs((s_prev - s)/s) < eps
            break
        end
        s_prev = s
        if mod(i, 100) == 0
            println("power iteration: $i $s")
        end
    end
    s
end

"""
    loop!(u, iterfun, evalfun, args...)

looping function
"""
function loop!(u, iterfun, evalfun, args...)
    converged = false
    t = 0
    while !converged && t < u.maxiter
        t += 1
        iterfun(args...)
        if t % u.step == 0
            converged, monitor = evalfun(args...)
            println("$(t)\t$(monitor)")
        end
    end
end

abstract type OptimConfig end
