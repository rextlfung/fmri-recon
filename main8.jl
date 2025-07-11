# %% Main8.jl
# Attempting to move away from using Pluto notebooks

# %% Import packages
# Math
using LinearAlgebra
using FFTW: fft!, ifft!, bfft!, fftshift!
using AbstractFFTs
using LinearMapsAA: LinearMapAA, block_diag, redim, undim
using MIRT: Afft, Asense, embed, pogm_restart, poweriter

# Probability and statistics
using Random: seed!
using Statistics, StatsBase

# Interpolation
using ImageTransformations

# Parallel computing
using Distributed
using Base.Threads

# Progress
using ProgressMeter

# Reading/writing files
using MAT, HDF5

# Plotting
using Plots
using MIRTjim: jim, prompt, mid3

# Readability
using Unitful: mm
using LaTeXStrings, Printf

# %% Include functionss
include("reconFuncs.jl")

# %% Declare and set path and experimental variables
# Path variables specific to this machine
top_dir = "/mnt/storage/rexfung/20250609ball/recon/"; # top directory
fn_ksp = top_dir * "47.mat"; # k-space file
fn_smaps = top_dir * "smaps.mat"; # sensitivity maps file
fn_recon_base = top_dir * "img47.mat"; # reconsctruced fMRI file

# %% Experimental parameters
# EPI parameters
N = (120, 120, 80) # Spatial tensor size
Nc = 32 # Number of coils
Nt = 60 # Number of time points
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

# %% Test multithreading
include("testMultithread.jl")
testMultithread()

# %% Load in zero-filled k-space data from .mat file
f_ksp = h5open(fn_ksp, "r") # opne file in read mode
ksp0 = f_ksp["ksp_epi_zf"][:, :, :, :, 1:Nt]
close(f_ksp)
ksp0 = Complex{Float32}[complex(k.real, k.imag) for k in ksp0]
@assert (N[1], N[2], N[3], Nc, Nt) == size(ksp0)
# %%
GC.gc()
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

# %% Plot sampling patterns
t = 1
jim(Ω[1, :, :, t]; colorbar=:none, title="Sampling patterns for frame $t. R ≈ $(round(R, sigdigits=4))", x=ky, xlabel=L"k_y", y=kz, ylabel=L"k_z")

# %% Plot cumulative sampling pattern
samp_sum = sum(Ω, dims=4)
color = cgrad([:blue, :black, :white], [0, 1 / 2Nt, 1])
jim(samp_sum[1, :, :]; color, clim=(0, Nt), title="Cumulative sampling pattern. R ≈ $(round(R, sigdigits=4))", x=ky, xlabel=L"k_y", y=kz, ylabel=L"k_z")

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
ksp0 = nothing
GC.gc()

# %% Set reconstruction hyperparameters
# Declare hyperparameters here to avoid scope issues
patch_size = [7, 7, 7] # side lengths for cubic patches
stride_size = [3, 3, 3] # strides in each direction when sweeping patches
λ_L = 5e-2 # weight for nuclear norm penalty term

# %% Compute Lipschitz constant of MRI forward operator
function poweriter2(
    A;
    niter::Int=200,
    tol::Real=1e-6,
    x0::AbstractArray{<:Number}=ones(eltype(A), size(A, 2)),
    chat::Bool=true,
)

    x = copy(x0)
    ratio_old = Inf
    @showprogress 1 "Power iterating..." for iter in 1:niter
        Ax = A * x
        ratio = norm(Ax) / norm(x)
        if abs(ratio - ratio_old) / ratio < tol
            chat && @info "done at iter $iter"
            break
        end
        ratio_old = ratio
        x = A' * Ax
        x /= norm(x)
    end
    return x, norm(A * x) / norm(x)
end

σ1A = nothing
σ1A = 0.9998373 # Computed from last time
if !(@isdefined σ1A) || isnothing(σ1A)
    (_, σ1A) = poweriter2(undim(A)) # Compute using power iteration. Takes ~14 mins
end

# %% Define cost functions, gradient, and step size
dc_cost = X -> 0.5 * norm(A * X - ksp)^2
nn_cost = X -> λ_L * patch_nucnorm(img2patches(X, patch_size, stride_size))
total_cost = X -> dc_cost(X) + nn_cost(X)

dc_cost_grad = X -> A' * (A * X - ksp) # gradient of data consistency term
μ = 1 / (σ1A^2) # step size for GD

# %% Initialize solution (zero-filled adjoint solution)
X0 = A' * ksp;

# %% Begin iterative reconstruction using ISTA (Otazo et al. 2015), without S part
Niters = 20
fn_recon = fn_recon_base[1:end-4] * "_$(Niters)itrs.mat"

# %%
if isfile(fn_recon)
    f_img = matread(fn_recon)
    X = f_img["X"]
    dc_costs = f_img["dc_costs"]
    nn_costs = f_img["nn_costs"]
else
    dc_costs = zeros(Niters + 1)
    nn_costs = zeros(Niters + 1)

    X = X0
    dc_costs[1] = dc_cost(X)
    nn_costs[1] = nn_cost(X)

    @showprogress 1 "Reconstructing..." for k in 1:Niters
        if k == 1 # Benchmark first iteration
            println("Computational metrics for one iteration:")
            # Gradient descent on data-consiannual progres stency
            @showtime X = X - μ * dc_cost_grad(X)
            # "Proximal" operator on nuclear norm
            @showtime X = patchSVST(X, λ_L, patch_size, stride_size)
        else
            X = X - μ * dc_cost_grad(X)
            X = patchSVST(X, λ_L, patch_size, stride_size)
        end

        # save costs
        dc_costs[k+1] = dc_cost(X)
        nn_costs[k+1] = nn_cost(X)
    end

    # save to file
    matwrite(fn_recon, Dict(
            "X" => X,
            "dc_costs" => dc_costs,
            "nn_costs" => nn_costs
        ); compress=true)
end

# %% Plot costs
plot(title="Optimization progress", xlabel="iteration", ylabel="cost")
plot!(0:Niters, dc_costs; label="data consistency", marker=:xcross)
plot!(0:Niters, nn_costs; label="nuclear norm penalty, λ_L = $λ_L", marker=:xcross)
plot!(0:Niters, dc_costs + nn_costs; label="overall cost", marker=:xcross)
plot!(legend=:best)
# %% Plot initial and final solutions
using GLMakie

frame = Observable(20)
plot(
    jim(mid3(@lift(X0[:, end:-1:1, end:-1:1, $frame])[]); title="|Zero-filled adjoint|", xlabel=L"x, z", ylabel=L"z, y"),
    jim(mid3(@lift(X[:, end:-1:1, end:-1:1, $frame])[]); title="|LLR recon|, λ_L = $λ_L", xlabel=L"x, z", ylabel=L"z, y"),
    layout=(1, 2),
    size=(1500, 650),
    sgtitle="Frame $(frame[]), Nx = $(N[1]), Ny = $(N[2]), Nz = $(N[3]), Nt = $Nt, R ≈ $(round(R, sigdigits=4))"
)


# %% Plot initial and final solutions
frame = 20
plot(
    jim(mid3(X0[:, end:-1:1, end:-1:1, frame]); title="|Zero-filled adjoint|", xlabel=L"x, z", ylabel=L"z, y"),
    jim(mid3(X[:, end:-1:1, end:-1:1, frame]); title="|LLR recon|, λ_L = $λ_L", xlabel=L"x, z", ylabel=L"z, y"),
    layout=(1, 2),
    size=(1500, 650),
    sgtitle="Frame $frame, Nx = $(N[1]), Ny = $(N[2]), Nz = $(N[3]), Nt = $Nt, R ≈ $(round(R, sigdigits=4))"
)

# %% Replot sampling mask
jim(Ω[1, :, :, frame]; colorbar=:none, title="Sampling patterns for frame $frame. R ≈ $(round(R, sigdigits=4))", x=ky, xlabel=L"k_y", y=kz, ylabel=L"k_z")

# %% Inspect patches of intitial and final solution
P0 = img2patches(X0, patch_size, stride_size)
Opnorms0 = mapslices(opnorm, P0, dims=(1, 2))
Opnorms0[Opnorms0.==0] .= eps()
P0 ./= Opnorms0
svs0 = mapslices(svdvals, P0, dims=(1, 2))
P0 = nothing

P = img2patches(X, patch_size, stride_size)
Opnorms = mapslices(opnorm, P, dims=(1, 2))
Opnorms[Opnorms.==0] .= eps()
P ./= Opnorms
svs = mapslices(svdvals, P, dims=(1, 2))
P = nothing

Np = size(svs, 3)

plot(title="Average SVs for $Np normalized patches", xlabel="SV index")
plot!(1:size(svs0, 1), mean(svs0, dims=3)[:, 1, 1]; label=L"X_0", marker=:xcross)
plot!(1:size(svs, 1), mean(svs, dims=3)[:, 1, 1]; label=L"X_∞", marker=:xcross)

# %%
GC.gc()
