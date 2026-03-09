#=
Collection of functions for conducting statistical analyses on reconstructued images.
fMRI stuff
=#
module utils

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

using Plots
"""
plotOpt(dc_costs::Vector, reg_costs::Vector, restarts::Vector, logscale::Bool=false)

Plot optimization progresses logged by POGM.

Inputs:
dc_costs: data consistency term.
reg_costs: regularization term.
restarts: bool vector indicating when momentum was restarted.

Optional inputs:
logscale: bool flag indicating log-scale plotting
"""
function plotOpt(dc_costs::Vector, reg_costs::Vector, restarts::AbstractVector, logscale::Bool=false)
    # 1. Define the iteration range
    Niters = length(dc_costs) - 1
    iters = 0:Niters

    # 2. Initialize the plot
    # Using 'legend=:topright' to fix the position
    plt = plot(iters, dc_costs, 
        label="Data Consistency", 
        xlabel="Iteration", 
        ylabel="Cost", 
        title="POGM Optimization Convergence", 
        lw=2,
        legend=:topright)

    plot!(plt, iters, reg_costs, 
        label="Regularizer", 
        lw=2)

    # Total cost is now a solid line
    plot!(plt, iters, dc_costs .+ reg_costs, 
        label="Total Cost", 
        lw=2, 
        linestyle=:solid, 
        color=:black)

    # 3. Handle Restarts with red dashed lines
    restart_iters = findall(restarts) .- 1

    if !isempty(restart_iters)
        # vline! accepts an array of x-coordinates
        vline!(plt, restart_iters, 
            label="Restart", 
            color=:red, 
            linestyle=:dash, 
            alpha=0.8)
    end

    # Log-scale plotting
    if logscale
        plot!(plt, yaxis=:log10)
        ylabel!("Cost (log-scale)")
    end

    display(plt)
end

end