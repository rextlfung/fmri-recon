#=
Collection of functions for conducting statistical analyses on reconstructued images.
fMRI stuff
=#
using Statistics: mean, std

"""
tSNR(img::AbstractArray)

Create tSNR maps on dynamic images.

Inputs:
img: N-dimensional complex time series data of size (..., Nt)

Outputs:
tSNR_map: N-dimensional tSNR map
"""
function tSNR(img::AbstractArray)
    mag = abs.(img)
    N = ndims(mag)
    ϵ = eps(eltype(mag)) # avoid dividing by 0
    return dropdims(mean(mag, dims=N) ./ (std(mag, dims=N) .+ ϵ); dims=N)
end