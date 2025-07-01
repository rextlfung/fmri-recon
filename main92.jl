# %% Main92.jl
# Like main91.jl but testing on my macbook
using Pkg
Pkg.activate(".")

# %% Import packages
# Linear algebra
using LinearAlgebra
using LinearMapsAA: LinearMapAA, block_diag, redim, undim
using MIRT: Asense

# Probability and statistics
using Statistics, StatsBase

# Interpolation
using ImageTransformations: imresize

# Parallel computing
using Base.Threads

# Progress
using ProgressMeter

# Reading/writing files
using MAT, HDF5

# Plotting
using Plots: plot, plot!
using MIRTjim: jim, mid3

# Readability
using Unitful: mm
using LaTeXStrings

# %% Local helper functions
include("testMultithread.jl")
include("reconFuncs.jl")
include("mirt_mod.jl");

# %% Declare and set path and experimental variables
# Path variables specific to this machine
top_dir = "/Users/rexfung/github/data/20241017tap/"; # top directory
fn_ksp = top_dir * "kdata2x3.mat"; # k-space file
fn_smaps = top_dir * "smaps.mat"; # sensitivity maps file
fn_recon_base = top_dir * "Xinf.mat"; # reconsctruced fMRI file

# %% Experimental parameters
# EPI parameters
N = (120, 120, 40) # Spatial tensor size
Nc = 32 # Number of coils
Nt = 20 # Number of time points
FOV = (216mm, 216mm, 72mm) # Field of view
Δ = FOV ./ N # Voxel size
kFOV = 2 ./ Δ # k-space field of view
Δk = 2 ./ FOV # k-space voxel size

# GRE parameters
N_gre = (108, 108, 108) # GRE image tensor size
FOV_gre = (216mm, 216mm, 216mm)

# %% Make k-space vectors for plotting
kx = (-(N[1] ÷ 2):(N[1]÷2-1)) .* Δk[1]
ky = (-(N[2] ÷ 2):(N[2]÷2-1)) .* Δk[2]
kz = (-(N[3] ÷ 2):(N[3]÷2-1)) .* Δk[3]

# %% Test multithreading
testMultithread()

# %% Load in zero-filled k-space data from .mat file
f_ksp = h5open(fn_ksp, "r") # opne file in read mode
start_frame = 4 # read in data after steady state is reached
ksp0 = f_ksp["ksp_zf"][:, :, :, :, start_frame:Nt]
Nt = size(ksp0, 5)
close(f_ksp)
ksp0 = Complex{Float32}[complex(k.real, k.imag) for k in ksp0]
@assert (N[1], N[2], N[3], Nc, Nt) == size(ksp0)
GC.gc()

# %% Infer sampling patterns from zero-filled k-space data
Ω = (ksp0[:, :, :, 1, :] .!= 0);

# %% Infer accleration/undersampling factor
R = prod(N) / sum(Ω[:, :, :, 1])

# %% Validate sampling patterns
# 1. All coils have the same sampling pattern
for ic in 2:Nc
    @assert Ω == (ksp0[:, :, :, ic, :] .!= 0)
    "Detected a different sampling pattern for coil $ic"
end

# 2. All time frames acquire the same number of samples
for it in 2:Nt
    @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :, :, it-1])
    "Detected a different number of samples for frame $it"
end

# %% Plot sampling patterns
t = 1
jim(Ω[1, :, :, t];
    colorbar=:none,
    title="Sampling patterns for frame $t. R ≈ $(round(R, sigdigits=4))",
    x=ky, xlabel=L"k_y",
    y=kz, ylabel=L"k_z")

# %% Plot cumulative sampling pattern
samp_sum = sum(Ω, dims=4)
color = cgrad([:blue, :black, :white], [0, 1 / 2Nt, 1])
jim(samp_sum[1, :, :];
    color, clim=(0, Nt),
    title="Cumulative sampling pattern. R ≈ $(round(R, sigdigits=4))",
    x=ky, xlabel=L"k_y",
    y=kz, ylabel=L"k_z")

# %% Load in sensitivity maps
smaps = matread(fn_smaps)["smaps"] # raw coil sensitivity maps

if size(smaps) !== size(ksp0)[1:4]
    # Crop to FOV
    x_range = round((FOV_gre[1] - FOV[1]) / FOV_gre[1] / 2 * N_gre[1] + 1)
              :round(N_gre[1] - (FOV_gre[1] - FOV[1]) / FOV_gre[1] / 2 * N_gre[1])
    y_range = round((FOV_gre[2] - FOV[2]) / FOV_gre[2] / 2 * N_gre[2] + 1)
              :round(N_gre[2] - (FOV_gre[2] - FOV[2]) / FOV_gre[2] / 2 * N_gre[2])
    z_range = round((FOV_gre[3] - FOV[3]) / FOV_gre[3] / 2 * N_gre[3] + 1)
              :round(N_gre[3] - (FOV_gre[3] - FOV[3]) / FOV_gre[3] / 2 * N_gre[3])
    smaps = smaps[Int.(x_range), Int.(y_range), Int.(z_range), :]

    # Interpolate to match EPI voxel sizes
    smaps_new = complex.(zeros(N[1], N[2], N[3], Nc));
    for coil = 1:Nc
        real_part = imresize(real(smaps[:, :, :, coil]), (N[1], N[2], N[3]))
        imag_part = imresize(imag(smaps[:, :, :, coil]), (N[1], N[2], N[3]))
        smaps_new[:, :, :, coil] = complex.(real_part, imag_part)
    end

    # Normalize sensitivity maps along the coil dimension
    smaps = smaps_new ./ sqrt.(sum(abs2.(smaps_new), dims=ndims(smaps_new)))
    smaps[isnan.(smaps)] .= 0
end

jim(mid3(smaps[:, :, :, Nc÷2]); title="Middle 3 planes of smaps for coil $(Nc ÷ 2)", xlabel=L"x, z", ylabel=L"z, y")

# %% SENSE forward model
# Otazo style MRI forward operator for a single time frame
Aotazo = (Ω, smaps) -> Asense(Ω, smaps; fft_forward=true, unitary=true)

# Encoding matrix for entire time series as block diagonal matrix
A = block_diag([Aotazo(s, smaps) for s in eachslice(Ω, dims=ndims(Ω))]...)

# Display input and output dimensions
println("Input dimensions: ", A._idim)
println("Output dimensions: ", A._odim)

# %% Preprocess k-space data to be in the shape of the odim of A
# Flatten spatial dimensions of k-space data and discard zeros
ksp = reshape(ksp0, :, Nc, Nt)
ksp = [ksp[vec(s), :, it] for (it, s) in enumerate(eachslice(Ω, dims=4))]
ksp = cat(ksp..., dims=3) # (Nsamples, Nc, Nt), no "zeros"
println("Shape of k-space data: ", size(ksp))
ksp0 = nothing;
GC.gc();

# %% Set reconstruction hyperparameters
# Declare hyperparameters here to avoid scope issues
patch_size = [6, 6, 6] # side lengths for cubic patches
stride_size = [3, 3, 3] # strides in each direction when sweeping patches
λ_L = 5e-2 # weight for nuclear norm penalty term

# %% Compute Lipschitz constant of MRI forward operator
σ1A = nothing
σ1A = 0.9998373 # Computed from last time
if !(@isdefined σ1A) || isnothing(σ1A)
    (_, σ1A) = poweriter_mod(undim(A)) # Compute using power iteration. Takes ~14 mins
end

# %% Define cost functions, gradient, and step size
dc_cost = X -> 0.5 * norm(A * X - ksp)^2 / norm(ksp)^2
nn_cost = X -> λ_L * patch_nucnorm(img2patches(X, patch_size, stride_size))
total_cost = X -> dc_cost(X) + nn_cost(X)
dc_cost_grad = X -> A' * (A * X - ksp) / norm(ksp)^2

# %% Initialize solution (zero-filled adjoint solution)
X0 = A' * ksp;

# %% Begin iterative reconstruction using ISTA (Otazo et al. 2015), without S part
Niters = 10
fn_recon = fn_recon_base[1:end-4] * "_$(Niters)itrs.mat"

# %%
if isfile(fn_recon)
    f_img = matread(fn_recon)
    X = f_img["X"]
    dc_costs = f_img["dc_costs"]
    nn_costs = f_img["nn_costs"]
else
    # Proximal step in the form compatible with pogm
    function g_prox(X, c)
        return patchSVST(X, λ_L, patch_size, stride_size)
    end

    # Log data-consistency and regularization costs
    logger = (iter, xk, yk, is_restart) -> (dc_cost(xk), nn_cost(xk))

    # Run POGM
    X, costs = pogm_mod(X0, (x) -> 0, dc_cost_grad, (σ1A/norm(ksp))^2;
        mom=:pogm, niter=Niters, g_prox=g_prox, fun=logger)

    # Unpack costs
    dc_costs = zeros(Niters + 1)
    nn_costs = zeros(Niters + 1)
    for i in 1:Niters+1
        dc_costs[i] = costs[i][1]
        nn_costs[i] = costs[i][2]
    end

    # Save to file
    matwrite(fn_recon, Dict(
            "X" => X,
            "dc_costs" => dc_costs,
            "nn_costs" => nn_costs
        ); compress=true)
end

# %% Plot costs over iterations
plot(0:Niters, dc_costs;
    label=L"\frac{1}{2} {||AX - y||}^2 / {||y||}^2",
    xlabel="iteration", ylabel="cost",
    marker=:xcross)
plot!(0:Niters, nn_costs;
    label=L"\lambda_L \sum_{p = 1}^{Np} {|| \mathcal{P}_p (X) ||}_*",
    xlabel="iteration", ylabel="cost",
    marker=:xcross)
title!("Optimization progress")

# %% Plot initial and final solutions
frame = 10
plot(
    jim(mid3(X0[:, end:-1:1, end:-1:1, frame]); title="|Zero-filled adjoint|", xlabel=L"x, z", ylabel=L"z, y"),
    jim(mid3(X[:, end:-1:1, end:-1:1, frame]); title="|LLR recon|, λ_L = $λ_L", xlabel=L"x, z", ylabel=L"z, y"),
    layout=(1, 2),
    size=(1500, 650),
    sgtitle="Frame $frame, Nx = $(N[1]), Ny = $(N[2]), Nz = $(N[3]), Nt = $Nt, R ≈ $(round(R, sigdigits=4))"
)

# %% Plot time series in the middle of the volume
plot(abs.(X0[N[1]÷2, N[2]÷2, N[3]÷2, :]), label=L"X_0")
plot!(abs.(X[N[1]÷2, N[2]÷2, N[3]÷2, :]), label=L"X_∞")
xlabel!("frame")
title!("Magnitude time series of voxel ($(N[1]÷2), $(N[2]÷2), $(N[3]÷2))")
# %% Plot time series in the middle of the volume
plot(abs.(X0[N[1]÷2+patch_size[1]+1, N[2]÷2+patch_size[2]+1, N[3]÷2, :]), label=L"X_0")
plot!(abs.(X[N[1]÷2+patch_size[1]+1, N[2]÷2+patch_size[2]+1, N[3]÷2, :]), label=L"X_∞")
xlabel!("frame")
title!("Magnitude time series of voxel ($(N[1]÷2 + patch_size[1]+1), $(N[2]÷2 + patch_size[2]+1), $(N[3]÷2))")

# %% Plot time series in the edge of the volume (should be pure noise)
plot(abs.(X0[60, 15, N[3]÷2, :]), label=L"X_0")
plot!(abs.(X[60, 15, N[3]÷2, :]), label=L"X_∞")
xlabel!("frame")
title!("Magnitude time series of voxel (60, 15, 40)")

# %% Plot time series in the edge of the volume (should be pure noise)
plot(abs.(X0[1, 1, 1, :]), label=L"X_0")
plot!(abs.(X[1, 1, 1, :]), label=L"X_∞")
xlabel!("frame")
title!("Magnitude time series of voxel (1, 1, 1)")

# %% Plot temporal mean
avg_0 = mean(X0, dims=4)
jim(avg_0; title="|Zero-filled adjoint temporal mean|", xlabel=L"x", ylabel=L"y")

# %%
avg_inf = mean(X, dims=4)
jim(avg_inf; title="|LLR recon temporal mean|", xlabel=L"x", ylabel=L"y")

# %% Plot temporal variance
var_0 = var(X0, dims=4)
jim(var_0; title="|Zero-filled adjoint temporal variance|", xlabel=L"x", ylabel=L"y")

# %%
var_inf = var(X, dims=4)
jim(var_inf; title="|LLR recon temporal variance|", xlabel=L"x", ylabel=L"y")
