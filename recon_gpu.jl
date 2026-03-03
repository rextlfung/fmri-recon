# recon_gpu.jl
module recon_gpu

using LinearAlgebra
using Statistics
using FFTW
using CUDA
using CUDA.CUFFT

# --- 3D Patching Functions ---

function img2patches(img::AbstractArray, patch_size, stride_size)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size

    psx, psy, psz = min(psx, Nx), min(psy, Ny), min(psz, Nz)

    Nsteps_x = fld(Nx - psx, ssx)
    Nsteps_y = fld(Ny - psy, ssy)
    Nsteps_z = fld(Nz - psz, ssz)
    Np = (Nsteps_x + 1) * (Nsteps_y + 1) * (Nsteps_z + 1)

    # Use similar() to ensure P is on the GPU if img is on the GPU
    P = fill!(similar(img, ComplexF32, (psx * psy * psz, Nt, Np)), 0)

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        # Slicing a CuArray returns a CuArray; assignment is handled as a GPU copy
        patch = img[ix*ssx.+(1:psx), iy*ssy.+(1:psy), iz*ssz.+(1:psz), :]
        P[:, :, ip] = reshape(patch, (psx * psy * psz, Nt))
        ip += 1
    end
    return P
end

function patches2img(P::AbstractArray, patch_size, stride_size, og_size)
    _, Nt, Np = size(P)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size
    Nx, Ny, Nz = og_size
    
    psx, psy, psz = min(psx, Nx), min(psy, Ny), min(psz, Nz)

    Nsteps_x = fld(Nx - psx, ssx)
    Nsteps_y = fld(Ny - psy, ssy)
    Nsteps_z = fld(Nz - psz, ssz)

    # Allocate output and count array on the same device as P
    img = fill!(similar(P, ComplexF32, (Nx, Ny, Nz, Nt)), 0)
    Pcount = fill!(similar(P, Float32, (Nx, Ny, Nz)), 0)

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        patch = reshape(P[:, :, ip], (psx, psy, psz, Nt))
        img[ix*ssx.+(1:psx), iy*ssy.+(1:psy), iz*ssz.+(1:psz), :] .+= patch
        Pcount[ix*ssx.+(1:psx), iy*ssy.+(1:psy), iz*ssz.+(1:psz)] .+= 1
        ip += 1
    end

    # Vectorized normalization
    Pcount .= max.(Pcount, 1.0f0) 
    img ./= reshape(Pcount, Nx, Ny, Nz, 1)

    return img
end

# --- Optimization Functions ---
function patch_nucnorm(P::AbstractArray)
    @assert ndims(P) == 3 "P should be a 3D tensor (space x time x patch)"

    Np = size(P, 3)
    
    # Pre-allocate the costs vector on the same device as P (CPU or GPU)
    # Using Float32 for typical GPU performance gains
    costs = fill!(similar(P, Float32, Np), 0.0f0)

    for ip in 1:Np
        P_view = @view P[:, :, ip]

        # CUDA.jl supports svdvals() on CuArrays directly
        svs = svdvals(P_view)
        
        if !isempty(svs) && svs[1] > 0
            # Calculate nuclear norm normalized by spectral norm (svs[1])
            # This is a vectorized operation on the GPU
            costs[ip] = sum(svs ./ svs[1])
        end
    end

    # sum(costs) is a GPU reduction; very fast.
    return sum(costs) / Float32(Np)
end

function SVST(X::AbstractMatrix, β)
    # CUDA.jl provides GPU-accelerated SVD
    U, s, V = svd(X)
    
    # Soft-thresholding without scalar indexing
    # We use Float32(β) to ensure type consistency on GPU
    s_thresh = @. max(s - Float32(β), 0.0f0)
    
    # Reconstruct. The zero-values in s_thresh handle the "keep" logic 
    # while allowing for full GPU-parallel matrix multiplication.
    return (U .* s_thresh') * V'
end

function patchSVST(img::AbstractArray, β, patch_size, stride_size)
    P = img2patches(img, patch_size, stride_size)
    Np = size(P, 3)

    # opnorm(P[:,:,ip]) in a loop is slow but safe from scalar errors.
    # We calculate σ1 for each patch to normalize.
    for ip in 1:Np
        σ1 = opnorm(P[:, :, ip])
        P[:, :, ip] ./= (σ1 + eps(Float32))
        
        # Apply SVST
        P[:, :, ip] = SVST(P[:, :, ip], β)
        
        # Re-normalize
        σ1_new = opnorm(P[:, :, ip])
        P[:, :, ip] ./= (σ1_new + eps(Float32))
        P[:, :, ip] .*= σ1
    end

    return patches2img(P, patch_size, stride_size, size(img)[1:3])
end

# --- Imaging Functions ---

function sense_comb(ksp::AbstractArray, smaps::AbstractArray)
    (Nx, Ny, Nz, Nc, Nt) = size(ksp)
    
    # Perform IFT (FFTW on CPU, CUFFT on GPU)
    # We ensure input is shifted correctly for the library
    img_mc = fftshift(ifft(ifftshift(ksp, (1,2,3)), (1, 2, 3)), (1,2,3))
    
    # Final combine step: sum(conj(S) * I) / sum(|S|^2)
    # This is fully vectorized across coils and space
    numerator = sum(conj.(smaps) .* img_mc, dims=4)
    denominator = sum(abs2.(smaps), dims=4) .+ eps(Float32)
    
    return dropdims(numerator ./ denominator; dims=4)
end

# --- Commented out fundamentally GPU-unfriendly functions ---
#=
function nn_viewshare(ksp::AbstractArray)
    # This function requires heavy scalar indexing and branching logic.
    # Recommendation: Move data to CPU using Array(ksp), run the CPU version,
    # then move back to GPU with CuArray(ksp_nn).
end
=#

# --- 2D Patching Support ---

function img2patches2D(img::AbstractArray, patch_size, stride_size)
    Nz, Ny, Nt = size(img)
    psz, psy = patch_size
    ssz, ssy = stride_size
    Nsteps_z, Nsteps_y = fld(Nz - psz, ssz), fld(Ny - psy, ssy)
    Np = (Nsteps_z + 1) * (Nsteps_y + 1)

    P = fill!(similar(img, ComplexF32, (psz * psy, Nt, Np)), 0)
    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        P[:, :, ip] = reshape(img[iz*ssz.+(1:psz), iy*ssy.+(1:psy), :], (psz * psy, Nt))
        ip += 1
    end
    return P
end

function patches2img2D(P::AbstractArray, patch_size, stride_size, og_size)
    _, Nt, Np = size(P)
    psz, psy = patch_size
    ssz, ssy = stride_size
    Nz, Ny = og_size

    img = fill!(similar(P, ComplexF32, (Nz, Ny, Nt)), 0)
    Pcount = fill!(similar(P, Float32, (Nz, Ny)), 0)

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        patch = reshape(P[:, :, ip], (psz, psy, Nt))
        img[iz*ssz.+(1:psz), iy*ssy.+(1:psy), :] .+= patch
        Pcount[iz*ssz.+(1:psz), iy*ssy.+(1:psy)] .+= 1
        ip += 1
    end
    Pcount .= max.(Pcount, 1.0f0)
    img ./= reshape(Pcount, Nz, Ny, 1)
    return img
end

end # module