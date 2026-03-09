# %% Main9993.jl
# main9991.jl but with true multi-scale low-rank DECOMPOSITION
using Pkg
Pkg.activate("."); Pkg.update(); Pkg.resolve();

# %% Import packages
# Linear algebra
using LinearAlgebra
using LinearMapsAA: LinearMapAA, block_diag, redim, undim
using MIRT: Asense

# Probability and statistics
using Statistics, StatsBase

# Interpolation
using ImageTransformations: imresize

# Progress
using ProgressMeter

# Reading/writing files
using MAT, HDF5

# Plotting
using Plots
using MIRTjim: jim, mid3
# change backened
plotlyjs()
# Set global defaults
default(
    size = (1400, 900), # Width and height in pixels
    dpi = 400          # Dots per inch (resolution)
)

# Readability
using Unitful: mm
using LaTeXStrings

# %% Local helper modules
using Revise
includet("mirt_mod.jl"); using .mirt_mod
includet("recon.jl"); using .recon: sense_comb, img2patches, patch_nucnorm, patchSVST
includet("utils.jl"); using .utils: tSNR, plotOpt

# %% Declare and set path and experimental variables
# Path variables specific to this machine
top_dir = "/StorageRAID/rexfung/20251106balltap/tap/recon/"; # top directory
fn_ksp = top_dir * "rand6x.mat"; # k-space file
fn_smaps = top_dir * "smaps_bart.mat"; # sensitivity maps file
fn_recon_base = top_dir * "recon.mat"; # reconsctruced fMRI file

# %% Experimental parameters
# EPI parameters
N = (90, 90, 60) # Spatial tensor size
(Nx, Ny, Nz) = N
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
smaps = smaps_new ./ (sqrt.(sum(abs2.(smaps_new), dims=ndims(smaps_new))) .+ eps());

# %% Load in zero-filled k-space data from .mat file
ksp0 = h5read(fn_ksp, "ksp_epi_zf")
ksp0 = ComplexF32.([complex(k.real, k.imag) for k in ksp0])
@assert (N[1], N[2], N[3], Nvc, Nt) == size(ksp0)

# Normalize ksp0
img0 = sense_comb(ksp0, smaps)
scale_factor = quantile(vec(abs.(img0)), 0.99)
ksp0 ./= max(scale_factor, eps(Float32));

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

for it in 2:Nt
    @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :,   :, it-1]) "Detected a different number of samples for frame $it"
end

# %% SENSE forward model
# Otazo style MRI forward operator for a single time frame
Aotazo = (Ω, smaps) -> Asense(Ω, smaps; fft_forward=true, unitary=true)

# Encoding matrix for entire time series as block diagonal matrix
A = block_diag([Aotazo(s, smaps) for s in eachslice(Ω, dims=ndims(Ω))]...)

# %% Preprocess k-space data to be in the shape of the odim of A
# Flatten spatial dimensions of k-space data and discard zeros
ksp = reshape(ksp0, :, Nvc, Nt)
ksp = [ksp[vec(s), :, it] for (it, s) in enumerate(eachslice(Ω, dims=4))]
ksp = cat(ksp..., dims=3) # (Nsamples, Nvc, Nt_block), no "zeros"
println("Shape of k-space data: ", size(ksp))

# %% Compute Lipschitz constant of MRI forward operator
σ1A = 1;
if !(@isdefined σ1A) || isnothing(σ1A)
    (_, σ1A) = poweriter_mod(undim(A)) # Compute using power iteration. Takes ~13 mins, converged at itr 157.
    print("σ1A = ", round(σ1A, digits=3))
end

# %% Set reconstruction hyperparameters
# side lengths for cubic patches
# patch_sizes = [[Nx, Ny, Nz]] # global low-rank
# patch_sizes = [[Nx, Ny, Nz], [1, 1, 1]] # low-rank + sparse
patch_sizes = [[10, 10, 10]] # local low-rank
# patch_sizes = [[90, 90, 60], # multiscale low-rank
#             [30, 30, 30],
#             [10, 10, 10],
#             [6, 6, 6],
#             [1, 1, 1]] # Fix recon code for enforcing sparsity (add an if-else)!!!

strides = patch_sizes # non-overlapping patches
# strides = [cld.(patch_size, 2) for patch_size in patch_sizes] # 1/2 overlapping patches

Nscales = size(patch_sizes, 1)

L = Nscales*(σ1A^2); # Lipschitz constant

# scale-specific regularization weights
# computed according to recommendations in Ong, Lustig 2016
λs = [(sqrt(prod(patch_sizes[k])) + sqrt(Nt) + sqrt(log(prod(N)*Nt/max(prod(patch_sizes[k]),Nt))))
      for k in 1:Nscales]

# %% Cost functions, gradients, and proximal operators
function dc_cost(X::AbstractArray)
    return 0.5 * norm(A * dropdims(sum(X, dims=ndims(X)), dims=ndims(X)) - ksp)^2
end

function dc_cost_grad(X::AbstractArray)
    return repeat(A' * (A * dropdims(sum(X, dims=ndims(X)), dims=ndims(X)) - ksp), outer = [1, 1, 1, 1, Nscales])
end

function reg_cost(X::AbstractArray)
    return sum(λs[k] * patch_nucnorm(img2patches(view(X,:,:,:,:,k), patch_sizes[k], strides[k])) for k in 1:Nscales)
end

function g_prox(X::AbstractArray, c)
    for k in 1:Nscales
        @views X[:,:,:,:,k] = patchSVST(view(X,:,:,:,:,k), c*λs[k], patch_sizes[k], strides[k])
    end
    return X
end

function comp_cost(X::AbstractArray)
    return dc_cost(X) + reg_cost(X)
end

# Log costs restarts over iterations
logger = (iter, xk, yk, is_restart) -> (dc_cost(xk), reg_cost(xk), is_restart);

# %% Initialize solution X0
# One component per scale
# Initialize global low-rank component as adjoint recon
# Initialize other components as zero
X0 = zeros(ComplexF32, Nx, Ny, Nz, Nt, Nscales);
X0[:,:,:,:,1] = A' * ksp; nothing

# Scale λs to data
# λs .*= dc_cost(X0)/max(reg_cost(X0), eps(Float32))

# %% POGM
X = copy(X0);
Niters = 50
X, costs = pogm_mod(X, comp_cost, dc_cost_grad, L;
    mom=:pogm, niter=Niters, g_prox=g_prox, fun=logger)

# Unpack costs
dc_costs = zeros(Niters + 1)
reg_costs = zeros(Niters + 1)
restarts = falses(Niters + 1)
for i in 1:Niters+1
    dc_costs[i] = costs[i][1]
    reg_costs[i] = costs[i][2]
    restarts[i] = costs[i][3]
end

# %% Plot costs and restarts w/ plotlyjs()
plotlyjs(); plotOpt(dc_costs, reg_costs, restarts, true)

# %%
gr(); jim(sum(X,dims=5)[:,:,:,1])

# %%
jim(X[:,:,30,1,:])

# %% Save to file
fn_recon = fn_recon_base[1:end-4] * "_$(Nscales)scales.mat"
matwrite(fn_recon, Dict(
    "X" => X, # final image
    "dc_costs" => dc_costs, # data consistency record
    "reg_costs" => reg_costs, # regularizer record
    "restarts" => restarts, # POGM restarts record
    "R" => R, # acceleration factor of k-space sampling
    "sigma1A" => σ1A, # spectral norm of system matrix A
    "L" => L, # lipschitz constant
    "Nscales" => Nscales, # number of scales for multi-scale low-rank
    "patch_sizes" => patch_sizes, # patch sizes
    "strides" => strides, # patch strides
    "lambdas" => λs, # regularization weights
    "Niters" => Niters, # number of POGM iterations
); compress=true)

# %% Load file (manual use)
vars = matread(fn_recon)

for (key, val) in vars
    @eval $(Symbol(key)) = $val
end