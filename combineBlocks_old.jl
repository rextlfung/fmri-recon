# %% Helper script for combining individually reconstruted time block of the image
using MAT, ProgressMeter

# %%
fn_base = "/mnt/storage/rexfung/20251024ball/recon/mslr/rand6x_0.005";
Nx = 90
Ny = 90
Nz = 60
Nt = 300
TR = 0.8 # seconds
T_block = 40 # seconds
Nt_block = Int(round(T_block / TR))
block_starts = Int.(1:Nt_block:Nt)

# %%
X = ComplexF32.(zeros(Nx, Ny, Nz, Nt))
dc_costs = []; nn_costs = [];
R = 0; Nscales = 0; Niters_inner = 0; λ_L = 0; patch_sizes = zeros(Nscales, 3); strides = zeros(Nscales, 3)
@showprogress 1 "Combining blocks..." for block in 1:length(block_starts)
    fn_block = fn_base * "_block$block.mat"
    f_block = matread(fn_block)

    X[:,:,:,block_starts[block]:Int(min(block_starts[block] + Nt_block - 1, Nt))] .= f_block["X"]

    if block == 1
        dc_costs = f_block["dc_costs"]
        nn_costs = f_block["nn_costs"]
        R = f_block["R"]
        Nscales = f_block["Nscales"]
        Niters_inner = f_block["Niters_inner"]
        λ_L = f_block["lambda_L"]
        patch_sizes = f_block["patch_sizes"]
        strides = f_block["strides"]
    else
        dc_costs += f_block["dc_costs"]
        nn_costs += f_block["nn_costs"]
    end
end

# %%
fn_recon = fn_base * ".mat"
matwrite(fn_recon, Dict(
            "X" => X,
            "dc_costs" => dc_costs,
            "nn_costs" => nn_costs,
            "R" => R,
            "Nscales" => Nscales,
            "Niters_inner" => Niters_inner,
            "lambda_L" => λ_L,
            "patch_sizes" => patch_sizes,
            "strides" => strides
        ); compress=true)