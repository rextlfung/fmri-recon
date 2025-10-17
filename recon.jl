#=
Collection of functions to be used on iterative image reconstruction.
=#

using Base.Threads
using LinearAlgebra
using Statistics
using FFTW

"""
img2patches(img::AbstractArray, patch_size, stride_size)

patch extractor: image -> (space x time) patches of image
using cubic patches for simplicity

Inputs:
img: 3D time series data of size (Nx, Ny, Nz, Nt)
patch_size: length 3 vector describing the side lengths of cubic patches
stride_size: length 3 vector describing the stride lengths between patches

Outputs:
P: 3D tensor of stack of (space x time) patch matrices. size = (prod(patch_size), Nt, Np)
"""
function img2patches(img::AbstractArray, patch_size, stride_size)
    # unpack inputs
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size

    # if patch size exceeds image size, set it to image size
    psx = min(psx, Nx)
    psy = min(psy, Ny)
    psz = min(psz, Nz)

    # calculate number of steps in each direction
    Nsteps_x = fld(Nx - psx, ssx)
    Nsteps_y = fld(Ny - psy, ssy)
    Nsteps_z = fld(Nz - psz, ssz)

    # calculate number of patches to extract
    Np = (Nsteps_x + 1) * (Nsteps_y + 1) * (Nsteps_z + 1)

    # preallocate output array to hold extracted patches
    P = zeros(ComplexF32, (psx * psy * psz, Nt, Np))

    # slide through L and extract patches
    ip = 1 # patch counter
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        patch = img[ix*ssx.+(1:psx), iy*ssy.+(1:psy), iz*ssz.+(1:psz), :]
        P[:, :, ip] = reshape(patch, (psx * psy * psz, Nt))

        ip += 1
    end

    return P
end

"""
patches2img(P::AbstractArray, og_size, stride_size)

patch recombinator: (space x time) patches of image -> image
using cubic patches for simplicity

Inputs:
P: 3D tensor of stack of (space x time) patch matrices. size = (prod(patch_size), Nt, Np)
og_size: length 3 vector describing the original image spatial dimensions
stride_size: length 3 vector describing the stride lengths between patches

Outputs:
img: 3D time series data of size (Nx, Ny, Nz, Nt). Computed via averaging overlapping patches

"""
function patches2img(P::AbstractArray, patch_size, stride_size, og_size)
    # unpack inputs
    _, Nt, Np = size(P)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size
    Nx, Ny, Nz = og_size
    
    # if patch size exceeds image size, set it to image size
    psx = min(psx, Nx)
    psy = min(psy, Ny)
    psz = min(psz, Nz)

    # calculate number of steps in each direction
    Nsteps_x = fld(Nx - psx, ssx)
    Nsteps_y = fld(Ny - psy, ssy)
    Nsteps_z = fld(Nz - psz, ssz)

    # preallocate memory for output
    img = zeros(ComplexF32, (Nx, Ny, Nz, Nt))

    # counter for computing average
    Pcount = zeros(Nx, Ny, Nz)

    # slide through patches and allocate them to original
    ip = 1 # patch counter
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        patch = reshape(P[:, :, ip], (psx, psy, psz, Nt))
        img[ix*ssx.+(1:psx), iy*ssy.+(1:psy), iz*ssz.+(1:psz), :] .+= patch

        Pcount[ix*ssx.+(1:psx), iy*ssy.+(1:psy), iz*ssz.+(1:psz)] .+= 1

        ip += 1
    end

    # prevent division by 0 error for voxels uncovered by any patch
    Pcount[Pcount.==0] .= 1

    # Divide each voxel by their number of contributing patches
    img ./= Pcount

    return img
end

"""
patch_nucnorm(P::AbstractArray)

compute the sum of nuclear norm of patches.
each patch is first normalized by its spectral norm.
final cost is normalized by the number of patches

Inputs:
P: 3D tensor of (space x time) patch matrices. size = (prod(patch_size), Nt, Np)

Outputs:
cost: scalar normalized nuclear norm penalty.
"""
function patch_nucnorm(P::AbstractArray)
    @assert ndims(P) == 3 "P should be a 3D tensor (space x time x patch)"

    Np = size(P, ndims(P))
    costs = zeros(Np)

    @threads for ip in 1:Np
        svs = svdvals(copy(P[:, :, ip]))
        if svs[1] > 0
            costs[ip] = sum(svs ./ svs[1])
        else
            costs[ip] = 0
        end
    end

    return sum(costs) / Np
end;

"""
SVST(X::AbstractArray, β)

Singular Value Soft-Thresholding
proximal operator to nuclear norm

Inputs:
X: matrix to be low-rankified via SVST. Complex
β: threshold hyperparameter

Outputs:
low-rankified version of X
"""
function SVST(X::AbstractMatrix, β)
    U, s, V = svd(X)
    sthresh = @. max(abs(s) - β, 0) * exp(1im * angle(s))
    keep = findall(!=(0), sthresh)
    return U[:, keep] * Diagonal(sthresh[keep]) * V[:, keep]'
end;

"""
patchSVST(img::AbstractArray, β, patch_size, stride_size)

apply SVST to in a patch-wise manner to an image
average of proximal operators to the nuclear norm

Inputs:
img: 3D time series data of size (Nx, Ny, Nz, Nt)
β: soft-thresholding threshold
patch_size: length 3 vector describing the side lengths of cubic patches
stride_size: length 3 vector describing the stride lengths between patches

Outputs:
patch-wise low-rankified version of img
"""
function patchSVST(img::AbstractArray, β, patch_size, stride_size)
    # extract patches
    P = img2patches(img, patch_size, stride_size)
    Np = size(P, ndims(P))

    # normalize each patch via division of their 2-norm (leading SV)
    σ1s = [opnorm(P[:, :, ip]) for ip in 1:Np]
    σ1s[σ1s.==0] .= eps() # avoid division by 0
    P ./= reshape(σ1s, 1, 1, :)

    # low-rankify each patch
    @threads for ip in 1:Np
        P[:, :, ip] = SVST(P[:, :, ip], β) # can this be done in-place for speed?
    end

    # rescale so leading SV = 1 before reverting normalization
    σ1s_tmp = [opnorm(P[:, :, ip]) for ip in 1:Np]
    σ1s_tmp[σ1s_tmp.==0] .= eps() # avoid division by 0
    P ./= reshape(σ1s_tmp, 1, 1, :)

    # revert normalization to preserve inter-patch contrast
    P .*= reshape(σ1s, 1, 1, :)

    # recombine patches into image
    img = patches2img(P, patch_size, stride_size, size(img)[1:3])

    return img
end;

"""
nn_viewshare(ksp::AbstractArray)

perform nearest-neighbor interpolation along the time dimension for each k-space location
unsampled k-space locations remain zero

Inputs:
ksp: 3D k-space multi-coil time series of size (Nx, Ny, Nz, Nc, Nt)

Outputs:
ksp_nn: nearest-neighbor interpolated 3D k-space multi-coil time series of size (Nx, Ny, Nz, Nc, Nt)
"""
function nn_viewshare(ksp::AbstractArray)
    (Nx, Ny, Nz, Nc, Nt) = size(ksp)
    ksp_nn = zero.(ksp)
    new_grid = 1:Nt
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        k_vec = ksp[i,j,k,:,:]
        old_grid = findall(!iszero, k_vec[1,:])
        if isempty(old_grid)
            continue
        end
        idxs = [argmin(abs.(old_grid .- loc)) for loc in new_grid]
        ksp_nn[i,j,k,:,:] = k_vec[:, idxs]
    end
    return ksp_nn
end

"""
sense_comb(ksp::AbstractArray, smaps::AbstractArray)

perform sensitivity map weighted IFT recon
used for initializing X0

Inputs:
ksp: 3D k-space multi-coil time series of size (Nx, Ny, Nz, Nc, Nt)

Outputs:
img: nearest-neighbor interpolated 3D k-space multi-coil time series of size (Nx, Ny, Nz, Nt)
"""
function sense_comb(ksp::AbstractArray, smaps::AbstractArray)
    (Nx, Ny, Nz, Nc, Nt) = size(ksp)
    img = zeros(eltype(ksp), Nx, Ny, Nz, Nt)

    img_mc = fftshift(ifft(ifftshift(ksp), (1, 2, 3))) # 3D IFT
    for t in 1:Nt
        numerator = sum(conj.(smaps) .* img_mc[:,:,:,:,t], dims=4)
        denominator = sum(abs2.(smaps), dims=4) .+ eps()
        img[:,:,:,t] = dropdims(numerator ./ denominator; dims=4)
    end
    return img
end
# %% Functions for 2D x time data

"""
img2patches2D(img::AbstractArray, patch_size, stride_size)

patch extractor: image -> (space x time) patches of image
using 2D patches across the z and y dimensions

Inputs:
img: 3D time series data of size (Nz, Ny, Nt)
patch_size: length 2 vector describing the side lengths of rectangular patches (z, y)
stride_size: length 2 vector describing the stride lengths between patches (z, y)

Outputs:
P: 3D tensor of stack of (space x time) patch matrices. size = (prod(patch_size), Nt, Np)
"""
function img2patches2D(img::AbstractArray, patch_size, stride_size)
    # unpack inputs
    Nz, Ny, Nt = size(img)
    psz, psy = patch_size
    ssz, ssy = stride_size

    # calculate number of steps in each direction
    Nsteps_z = fld(Nz - psz, ssz)
    Nsteps_y = fld(Ny - psy, ssy)

    # total number of patches
    Np = (Nsteps_z + 1) * (Nsteps_y + 1)

    # preallocate output array
    P = zeros(ComplexF32, (psz * psy, Nt, Np))

    # extract patches
    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        patch = img[iz*ssz.+(1:psz), iy*ssy.+(1:psy), :]
        P[:, :, ip] = reshape(patch, (psz * psy, Nt))
        ip += 1
    end

    return P
end

"""
patches2img2D(P::AbstractArray, patch_size, stride_size, og_size)

patch recombinator: (space x time) patches of image -> image
using 2D patches across the z and y dimensions

Inputs:
P: 3D tensor of stack of (space x time) patch matrices. size = (prod(patch_size), Nt, Np)
patch_size: length 2 vector describing the side lengths of rectangular patches (z, y)
stride_size: length 2 vector describing the stride lengths between patches (z, y)
og_size: length 2 vector describing the original image spatial dimensions (Nz, Ny)

Outputs:
img: 3D time series data of size (Nz, Ny, Nt). Computed via averaging overlapping patches
"""
function patches2img2D(P::AbstractArray, patch_size, stride_size, og_size)
    # unpack inputs
    _, Nt, Np = size(P)
    psz, psy = patch_size
    ssz, ssy = stride_size
    Nz, Ny = og_size

    # calculate number of steps
    Nsteps_z = fld(Nz - psz, ssz)
    Nsteps_y = fld(Ny - psy, ssy)

    # preallocate output image and count array
    img = zeros(ComplexF32, (Nz, Ny, Nt))
    Pcount = zeros(Nz, Ny)

    # place patches into image
    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        patch = reshape(P[:, :, ip], (psz, psy, Nt))
        img[iz*ssz.+(1:psz), iy*ssy.+(1:psy), :] .+= patch
        Pcount[iz*ssz.+(1:psz), iy*ssy.+(1:psy)] .+= 1
        ip += 1
    end

    # avoid division by zero
    Pcount[Pcount.==0] .= 1

    # normalize by overlap count
    for t in 1:Nt
        img[:, :, t] ./= Pcount
    end

    return img
end

"""
patchSVST2D(img::AbstractArray, β, patch_size, stride_size)

apply SVST in a patch-wise manner to a 2D x time image
average of proximal operators to the nuclear norm

Inputs:
img: 3D time series data of size (Nz, Ny, Nt)
β: soft-thresholding threshold
patch_size: length 2 vector describing the side lengths of rectangular patches (z, y)
stride_size: length 2 vector describing the stride lengths between patches (z, y)

Outputs:
patch-wise low-rankified version of img
"""
function patchSVST2D(img::AbstractArray, β, patch_size, stride_size)
    # extract patches
    P = img2patches2D(img, patch_size, stride_size)
    Np = size(P, ndims(P))

    # normalize each patch by dividing by its leading singular value
    σ1s = [opnorm(P[:, :, ip]) for ip in 1:Np]
    σ1s[σ1s.==0] .= eps() # avoid division by 0
    P ./= reshape(σ1s, 1, 1, :)

    # low-rankify each patch using SVST
    @threads for ip in 1:Np
        P[:, :, ip] = SVST(P[:, :, ip], β)  # not in-place due to SVD limitation
    end

    # re-normalize to leading singular value = 1
    σ1s_tmp = [opnorm(P[:, :, ip]) for ip in 1:Np]
    σ1s_tmp[σ1s_tmp.==0] .= eps()
    P ./= reshape(σ1s_tmp, 1, 1, :)

    # scale back to original patch magnitudes
    P .*= reshape(σ1s, 1, 1, :)

    # recombine patches into image
    return patches2img2D(P, patch_size, stride_size, size(img)[1:2])
end
