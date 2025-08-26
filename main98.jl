# %% Main98.jl
# Retrospectively undersampled data from fully-sampled GRE images
using Pkg
Pkg.activate(".")

# %% Import packages
# Linear algebra
using LinearAlgebra
using LinearMapsAA: LinearMapAA, block_diag, redim, undim
using MIRT: Asense, pogm_restart

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
using Plots
using MIRTjim: jim, mid3

# Readability
using Unitful: mm
using LaTeXStrings

# %% Local helper functions
include("mirt_mod.jl")
include("recon.jl")
include("analysis.jl")

# %% Declare and set path and experimental variables
# Path variables specific to this machine
top_dir = "/mnt/storage/rexfung/20250819retroball/"; # top directory
fn_ksp = top_dir * "ksp_us_6x.mat"; # k-space file
fn_smaps = top_dir * "smaps.mat"; # sensitivity maps file
fn_recon_base = top_dir * "recon.mat"; # reconsctruced fMRI file

# %% Experimental parameters
# EPI parameters
N = (108, 108, 60) # Spatial tensor size
Nt = 50 # Number of time points
Nc = 10 # Number of virtual coils
FOV = (216mm, 216mm, 120mm) # Field of view
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

# %% Load in zero-filled k-space data from .mat file
f_ksp = h5open(fn_ksp, "r") # opne file in read mode
start_frame = 1 # read in data after steady state is reached
ksp0 = f_ksp["ksp_us_mc"][:, :, :, :, start_frame:Nt]
Nt = size(ksp0, 5)
close(f_ksp)
ksp0 = Complex{Float32}[complex(k.real, k.imag) for k in ksp0]
@assert (N[1], N[2], N[3], Nc, Nt) == size(ksp0)

# %% Infer sampling patterns from zero-filled k-space data
    Ω = (ksp0[:, :, :, 1, :] .!= 0);

# %% Infer accleration/undersampling factor
R = prod(N) / sum(Ω[:, :, :, 1])

# %% Validate sampling patterns
# 1. All coils have the same sampling pattern
for ic in 2:Nc
    @assert Ω == (ksp0[:, :, :, ic, :] .!= 0) "Detected a different sampling pattern for coil $ic"
end

# 2. All time frames acquire the same number of samples
for it in 2:Nt
    @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :, :, it-1]) "Detected a different number of samples for frame $it"
end

# %% Load in sensitivity maps
smaps = matread(fn_smaps)["smaps_raw"] # raw coil sensitivity maps

# Crop to FOV
x_range = round((FOV_gre[1] - FOV[1]) / FOV_gre[1] / 2 * N_gre[1] + 1):round(N_gre[1] - (FOV_gre[1] - FOV[1]) / FOV_gre[1] / 2 * N_gre[1])
y_range = round((FOV_gre[2] - FOV[2]) / FOV_gre[2] / 2 * N_gre[2] + 1):round(N_gre[2] - (FOV_gre[2] - FOV[2]) / FOV_gre[2] / 2 * N_gre[2])
z_range = round((FOV_gre[3] - FOV[3]) / FOV_gre[3] / 2 * N_gre[3] + 1):round(N_gre[3] - (FOV_gre[3] - FOV[3]) / FOV_gre[3] / 2 * N_gre[3])
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

# %% SENSE forward model
# Otazo style MRI forward operator for a single time frame
Aotazo = (Ω, smaps) -> Asense(Ω, smaps; fft_forward=true, unitary=true)

# Encoding matrix for entire time series as block diagonal matrix
A = block_diag([Aotazo(s, smaps) for s in eachslice(Ω, dims=ndims(Ω))]...)

# %% Preprocess k-space data to be in the shape of the odim of A
# Flatten spatial dimensions of k-space data and discard zeros
ksp = reshape(ksp0, :, Nc, Nt)
ksp = [ksp[vec(s), :, it] for (it, s) in enumerate(eachslice(Ω, dims=4))]
ksp = cat(ksp..., dims=3) # (Nsamples, Nc, Nt), no "zeros"
println("Shape of k-space data: ", size(ksp))

# %% Compute Lipschitz constant of MRI forward operator
σ1A = nothing
σ1A = 0.999 # Approximately
if !(@isdefined σ1A) || isnothing(σ1A)
    (_, σ1A) = poweriter_mod(undim(A)) # Compute using power iteration. Takes ~20 mins, converged at itr 125.
end

# %% Define dc cost and its gradient
dc_cost = X -> 0.5 * norm(A * X - ksp)^2 / norm(ksp)^2
dc_cost_grad = X -> A' * (A * X - ksp) / norm(ksp)^2

# %% Initialize solution
# X0 = A' * ksp; # zero-filled
X0 = repeat(mean(A' * ksp, dims = 4), outer = [1, 1, 1, Nt]); # temporal average
# nearest-neighbor interpolated, smaps-weighted IFT recon
# ksp_nn = nn_viewshare(ksp0)
# X0 = sense_comb(ksp_nn, smaps)

# %% Begin iterative reconstruction using ISTA (Otazo et al. 2015), without S part
Niters_outer = 3 # Number of outer iterations, each using a different proximal operator
Niters_inner = 10 # Number of inner iterations, each using the same proximal operator
Niters = Niters_outer * Niters_inner
fn_recon = fn_recon_base[1:end-4] * "_$(Niters)itrs.mat"

if isfile(fn_recon)
    f_img = matread(fn_recon)
    X = f_img["X"]
    dc_costs = f_img["dc_costs"]
    nn_costs = f_img["nn_costs"]
    λ_L = f_img["lambda_L"]
    Niters_inner = f_img["Niters_inner"]
    Niters_outer = f_img["Niters_outer"]
    R = f_img["R"]
else
    # Set reconstruction hyperparameters
    patch_size = (16, 16, 16) .* 2^(Niters_outer - 1) # side lengths for cubic patches
    stride_size = (6, 6, 6) # strides in each direction when sweeping patches
    λ_L = 5e-3 # weight for nuclear norm penalty term. Also represents the threshold of discarded SVs at every inner iteration

    # Define first regularizer as global nuclear norm
    nn_cost = X -> λ_L * patch_nucnorm(img2patches(X, patch_size, stride_size))

    # Compute initial cost
    dc_costs = zeros(Niters+1)
    nn_costs = zeros(Niters+1)
    dc_costs[1] = dc_cost(X0)
    nn_costs[1] = nn_cost(X0)

    # Iterate
    X = X0
    for k in 1:Niters_outer
        println("Reconstructing outer iteration $k / $Niters_outer.
                patch_size = $patch_size, stride = $stride_size")
        
        # Redefine nuclear norm and proximal step
        global nn_cost = X -> λ_L * patch_nucnorm(img2patches(X, patch_size, stride_size))
        function g_prox(X, c)
            return patchSVST(X, λ_L, patch_size, stride_size)
        end

        # Log data-consistency and regularization costs
        logger = (iter, xk, yk, is_restart) -> (dc_cost(xk), nn_cost(xk))

        # POGM
        global X, costs = pogm_mod(X, (x) -> 0, dc_cost_grad, (σ1A / norm(ksp))^2;
            mom=:pogm, niter=Niters_inner, g_prox=g_prox, fun=logger)

        # Save costs
        iter_range = 1 .+ ((1 + (k-1)*Niters_inner):k*Niters_inner)
        for l in 1:Niters_inner
            global dc_costs[iter_range[l]] = costs[1 + l][1]
            global nn_costs[iter_range[l]] = costs[1 + l][2]
        end

        # Reduce patch size for next outer iteration
        global patch_size = Int.(round.(patch_size .* (1/2)))
    end

    # Save to file
    matwrite(fn_recon, Dict(
            "X" => X,
            "dc_costs" => dc_costs,
            "nn_costs" => nn_costs,
            "lambda_L" => λ_L,
            "Niters_inner" => Niters_inner,
            "Niters_outer" => Niters_outer,
            "R" => R
            ); compress=true)
end

