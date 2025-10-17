# %% Helper script for combining individually reconstruted time block of the image
using MAT, ProgressMeter

# %%
fn_base = "/mnt/storage/rexfung/20251003tap/recon/mslr/6x_0.0025";
Nx = 90
Ny = 90
Nz = 60
Nt = 438
TR = 0.8 # seconds
T_block = 40 # seconds
Nt_block = Int(round(T_block / TR))
block_starts = Int.(1:Nt_block:Nt)

# %%
X = ComplexF32.(zeros(Nx, Ny, Nz, Nt))
@showprogress 1 "Combining blocks..." for block in 1:length(block_starts)
    fn_block = fn_base * "_block$block.mat"
    X[:,:,:,block_starts[block]:Int(min(block_starts[block] + Nt_block - 1, Nt))] .= matread(fn_block)["X"]
end

# %%
fn_recon = fn_base * ".mat"
matwrite(fn_recon, Dict("X" => X); compress=true)