"""
    gather!(out, vec, ind)

Simply returns out .= vec[ind]. Intended for CuArrays.
"""
function gather!(out, vec, ind)
    out .= vec[ind]
end