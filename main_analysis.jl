# %% main_analysis.jl
# Seperate main script for loading in data and analysing them
using Pkg; Pkg.activate(".")

# %% Import packages
using LinearAlgebra
using Statistics
using MAT
using Plots
using MIRTjim
using Unitful: mm
using LaTeXStrings

# %% Helper functions
include("analysis.jl")

# %% Read in recon
fn_recon = "/mnt/storage/rexfung/20250609ball/recon/img47_50itrs_5e-2.mat"
f_img = matread(fn_recon)
X = f_img["X"]
R = f_img["R"]
dc_costs = f_img["dc_costs"]
nn_costs = f_img["nn_costs"]
Niters_outer = f_img["Niters_outer"]
Niters_inner = f_img["Niters_inner"]
Niters = Niters_inner*Niters_outer
λ_L = f_img["lambda_L"]
patch_sizes = f_img["patch_sizes"]
strides = f_img["strides"]
display(f_img)

# %% Detrend data (remove linear drift)
(Nx, Ny, Nz, Nt) = size(X)
M = [ones(Nt) 1:Nt] # design matrix
Mpinv = pinv(M)
for i in 1:Nx, j in 1:Ny, k in 1:Nz
    β = Mpinv * vec(X[i,j,k,:])
    X[i,j,k,:] .-= M[:,2] * β[2] # only remove linear component
end

# %% Plot costs over iterations
p = plot(0:Niters, dc_costs;
    label=L"\frac{1}{2} \frac{{||\mathcal{A}(X) - Y||}_F^2}{{||Y||}_F^2}",
    xlabel="iteration", ylabel="cost",
    marker=:xcross,
    legendfontsize=16)
plot!(0:Niters, nn_costs;
    label=L"\frac{\lambda_L}{N_p} \sum_{p = 1}^{N_p} \frac{{|| \mathcal{P}(X)_p ||}_*}{{|| \mathcal{P}(X)_p ||}_2}",
    xlabel="iteration", ylabel="cost",
    marker=:xcross)
plot!(0:Niters, dc_costs .+ nn_costs;
    label="Total cost",
    xlabel="iteration", ylabel="cost",
    marker=:xcross,
    legend=:outerright)
title!("Multiscale LLR optimization progress. R = $(round(R, digits=2)).")
hline!([λ_L], color=:red, linestyle=:dash, label=L"λ_L" *" = $λ_L")
xticks = 0:Niters_inner:Niters
xticks!(p, xticks)
vline!(xticks, label="", linestyle=:dash, color=:gray)
ypos = maximum(dc_costs .+ nn_costs)  # slightly above the plot
for i in 1:Niters_outer
    xpos = (xticks[i] + xticks[i+1]) / 2
    txt = "patch size = $(patch_sizes[i])\nstride = $(strides[i])"
    annotate!(xpos, ypos, text(txt, :black, 9, :center))
end
plot!()

# %% Plot tSNR maps
tSNR_map = tSNR(X)
tSNR_map_masked = filter(x -> x > 0, tSNR_map)
mean_tSNR = mean(vec(tSNR_map_masked))
peak_tSNR = maximum(vec(tSNR_map_masked))
jim(tSNR_map; xlabel=L"x", ylabel=L"y", color=:inferno,
    title="tSNR stats: mean = $(round(mean_tSNR, digits=2)), peak = $(round(peak_tSNR, digits=2))")

# %% Plot histogram of tSNRs
# histogram(filter(x -> x > 0, tSNR_map),
#           bins = 100,
#           xlabel = "tSNR",
#           ylabel = "Voxel count",
#           title = "Histogram of tSNR values, mean = $(round(mean_tSNR, digits=2)), peak = $(round(peak_tSNR, digits=2))")
