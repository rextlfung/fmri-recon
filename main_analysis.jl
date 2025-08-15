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
fn_recon = "/mnt/storage/rexfung/20250609ball/recon/img45_15itrs_msllr.mat"
f_img = matread(fn_recon)
X = f_img["X"]
dc_costs = f_img["dc_costs"]
nn_costs = f_img["nn_costs"]
λ_L = f_img["lambda_L"]
Niters_inner = f_img["Niters_inner"]
Niters_outer = f_img["Niters_outer"]
display(f_img)

# %% Detrend data (remove linear drift)
(Nx, Ny, Nz, Nt) = size(X)
M = [ones(Nt) 1:Nt] # design matrix
Mpinv = pinv(M)
for i in 1:Nx, j in 1:Ny, k in 1:Nz
    β = Mpinv * vec(X[i,j,k,:])
    X[i,j,k,:] .-= M[:,2] * β[2] # only remove linear component
end

# %% Plot tSNR maps
tSNR_map = tSNR(X)
tSNR_map_masked = filter(x -> x > 0, tSNR_map)
mean_tSNR = mean(vec(tSNR_map_masked))
peak_tSNR = maximum(vec(tSNR_map_masked))
jim(tSNR_map; xlabel=L"x", ylabel=L"y", color=:inferno,
    title="tSNR stats: mean = $(round(mean_tSNR, digits=2)), peak = $(round(peak_tSNR, digits=2))")

# %% Plot histogram of tSNRs
histogram(filter(x -> x > 0, tSNR_map),
          bins = 100,
          xlabel = "tSNR",
          ylabel = "Voxel count",
          title = "Histogram of tSNR values, mean = $(round(mean_tSNR, digits=2)), peak = $(round(peak_tSNR, digits=2))")