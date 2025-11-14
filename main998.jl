# %% Main998.jl
# 20251106 ball + tap data. 2.4mm. Local in time. Cycle through patch sizes per iteration
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
top_dir = "/mnt/storage/rexfung/20251106balltap/tap/"; # top directory
fn_ksp = top_dir * "rand6x.mat"; # k-space file
fn_smaps = top_dir * "smaps_bart.mat"; # sensitivity maps file
fn_recon_base = top_dir * "mslr/rand6x.mat"; # reconsctruced fMRI file

# %% Experimental parameters
# EPI parameters
N = (90, 90, 60) # Spatial tensor size
Nvc = 18 # Number of virtual coils
Nt = 300 # Number of time points
FOV = (216mm, 216mm, 144mm) # Field of view
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

# %% Load in sensitivity maps
smaps = matread(fn_smaps)["smaps_raw"] # raw coil sensitivity maps

# Crop to FOV
x_range = round((FOV_gre[1] - FOV[1]) / FOV_gre[1] / 2 * N_gre[1] + 1):round(N_gre[1] - (FOV_gre[1] - FOV[1]) / FOV_gre[1] / 2 * N_gre[1])
y_range = round((FOV_gre[2] - FOV[2]) / FOV_gre[2] / 2 * N_gre[2] + 1):round(N_gre[2] - (FOV_gre[2] - FOV[2]) / FOV_gre[2] / 2 * N_gre[2])
z_range = round((FOV_gre[3] - FOV[3]) / FOV_gre[3] / 2 * N_gre[3] + 1):round(N_gre[3] - (FOV_gre[3] - FOV[3]) / FOV_gre[3] / 2 * N_gre[3])
smaps = smaps[Int.(x_range), Int.(y_range), Int.(z_range), :]

# Interpolate to match EPI voxel sizes
smaps_new = complex.(zeros(N[1], N[2], N[3], Nvc));
for coil = 1:Nvc
    real_part = imresize(real(smaps[:, :, :, coil]), (N[1], N[2], N[3]))
    imag_part = imresize(imag(smaps[:, :, :, coil]), (N[1], N[2], N[3]))
    smaps_new[:, :, :, coil] = complex.(real_part, imag_part)
end

# Normalize sensitivity maps tensor by its Frobenius norm
smaps = smaps_new ./ (sqrt.(sum(abs2.(smaps_new), dims=ndims(smaps_new))) .+ eps())

# %% Load in blocks of the experiment to recon with a smaller memory footprint
TR = 0.8 # seconds
T_block = 40 # seconds
Nt_block = Int(round(T_block / TR))
block_starts = Int.(1:Nt_block:Nt)
for block in 1:length(block_starts)
    # %% Load in zero-filled k-space data from .mat file
    f_ksp = h5open(fn_ksp, "r") # opne file in read mode
    ksp0 = f_ksp["ksp_epi_zf"][:, :, :, :, block_starts[block]:Int(min(block_starts[block] + Nt_block - 1, Nt))]
    global Nt_block = size(ksp0, 5)
    close(f_ksp)
    ksp0 = Complex{Float32}[complex(k.real, k.imag) for k in ksp0]
    @assert (N[1], N[2], N[3], Nvc, Nt_block) == size(ksp0)

    # %% Infer sampling patterns from zero-filled k-space data
    Ω = (ksp0[:, :, :, 1, :] .!= 0);

    # %% Infer accleration/undersampling factor
    R = prod(N) / sum(Ω[:, :, :, 1])

    # %% Validate sampling patterns
    # 1. All coils have the same sampling pattern
    for ic in 2:Nvc
        @assert Ω == (ksp0[:, :, :, ic, :] .!= 0) "Detected a different sampling pattern for coil $ic"
    end

    # 2. All time frames acquire the same number of samples
    for it in 2:Nt_block
        @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :,   :, it-1]) "Detected a different number of samples for frame $it"
    end

    # %% SENSE forward model
    # Otazo style MRI forward operator for a single time frame
    Aotazo = (Ω, smaps) -> Asense(Ω, smaps; fft_forward=true, unitary=true)

    # Encoding matrix for entire time series as block diagonal matrix
    A = block_diag([Aotazo(s, smaps) for s in eachslice(Ω, dims=ndims(Ω))]...)

    # %% Preprocess k-space data to be in the shape of the odim of A
    # Flatten spatial dimensions of k-space data and discard zeros
    ksp = reshape(ksp0, :, Nvc, Nt_block)
    ksp = [ksp[vec(s), :, it] for (it, s) in enumerate(eachslice(Ω, dims=4))]
    ksp = cat(ksp..., dims=3) # (Nsamples, Nvc, Nt_block), no "zeros"
    println("Shape of k-space data: ", size(ksp))

    # %% Compute Lipschitz constant of MRI forward operator
    σ1A = nothing
    # σ1A = 1.0 # 20251003tap rand6x
    # σ1A = 0.892 # 20251024ball rand6x
    σ1A = 1 # 20251106 rand6x
    if !(@isdefined σ1A) || isnothing(σ1A)
        (_, σ1A) = poweriter_mod(undim(A)) # Compute using power iteration. Takes ~20 mins, converged at itr 125.
        print("σ1A = ", round(σ1A, digits=3))
    end

    # %% Initialize solution
    X0 = A' * ksp; # zero-filled
    X0 = repeat(mean(X0, dims = 4), outer = [1, 1, 1, Nt_block]); # temporal average

    # %% Recon for a variety of parameters
    X = X0

    # Set reconstruction hyperparameters for each scale
    # side lengths for cubic patches
    patch_sizes = [[90, 90, 60],
                [45, 45, 30],
                [30, 30, 30],
                [15, 15, 15],
                [10, 10, 10],
                [6, 6, 6],
                [3, 3, 3],
                [1, 1, 1]]
    strides = patch_sizes # non-overlapping patches
    # patch_sizes = [[6, 6, 6]]
    # strides = [[3, 3, 3]]

    # weight for nuclear norm penalty term. Also represents the threshold of discarded SVs at every inner iteration
    λ_L = 5e-3

    # %% Define dc cost and its gradient
    dc_cost = X -> 0.5 * norm(A * X - ksp)^2
    dc_cost_grad = X -> A' * (A * X - ksp)

    Nscales = size(patch_sizes, 1) # Number of scales, each using a different proximal operator

    # %% Define nn cost and its proximal step
    function nn_cost(X)
        cost = 0
        for k in 1:Nscales
            cost += λ_L * patch_nucnorm(img2patches(X, patch_sizes[k], strides[k]))
        end
        return cost / Nscales
    end

    function g_prox(X, c)
        println()
        println("Threshold (c × λ_L) = ", round(c*λ_L, digits=3))
        for k in 1:Nscales
            X = patchSVST(X, c*λ_L, patch_sizes[k], strides[k])
        end
        return X
    end

    # Log data-consistency and regularization costs
    logger = (iter, xk, yk, is_restart) -> (dc_cost(xk), nn_cost(xk), is_restart)

    # POGM
    Niters = 20
    X, costs = pogm_mod(X, (X) -> dc_cost(X) + nn_cost(X), dc_cost_grad, σ1A^2;
        mom=:pogm, niter=Niters, g_prox=g_prox, fun=logger)

    # Unpack costs
    dc_costs = zeros(Niters + 1)
    nn_costs = zeros(Niters + 1)
    restarts = falses(Niters + 1)
    for i in 1:Niters+1
        dc_costs[i] = costs[i][1]
        nn_costs[i] = costs[i][2]
        restarts[i] = costs[i][3]
    end

    # Save to file
    fn_recon = fn_recon_base[1:end-4] * "_$(λ_L)_block$block.mat"
    matwrite(fn_recon, Dict(
            "X" => X,
            "R" => R,
            "dc_costs" => dc_costs,
            "nn_costs" => nn_costs,
            "restarts" => restarts,
            "Nscales" => Nscales,
            "Niters" => Niters,
            "lambda_L" => λ_L,
            "patch_sizes" => patch_sizes,
            "strides" => strides,
            "sigma1A" => σ1A
        ); compress=true)
end