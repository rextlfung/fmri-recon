#=
Collection of functions to be used on iterative image reconstruction.
=#

using Base.Threads
using LinearAlgebra

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

    # calculate number of steps in each direction
    Nsteps_x = fld(Nx - psx, ssx)
    Nsteps_y = fld(Ny - psy, ssy)
    Nsteps_z = fld(Nz - psz, ssz)

    # calculate number of patches to extract
    Np = (Nsteps_x + 1)*(Nsteps_y + 1)*(Nsteps_z + 1)

    # preallocate output array to hold extracted patches
    P = zeros(ComplexF32, (psx*psy*psz, Nt, Np))

    # slide through L and extract patches
    ip = 1 # patch counter
    for iz in 0:Nsteps_z
        for iy in 0:Nsteps_y
            for ix in 0:Nsteps_x
                patch = img[ix*ssx.+(1:psx),iy*ssy.+(1:psy),iz*ssz.+(1:psz),:]
                P[:,:,ip] = reshape(patch, (psx*psy*psz, Nt))

                ip += 1
            end
        end
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
    for iz in 0:Nsteps_z
        for iy in 0:Nsteps_y
            for ix in 0:Nsteps_x
                patch = reshape(P[:,:,ip], (psx, psy, psz, Nt))
                img[ix*ssx.+(1:psx),iy*ssy.+(1:psy),iz*ssz.+(1:psz), :] .+= patch

                Pcount[ix*ssx.+(1:psx),iy*ssy.+(1:psy),iz*ssz.+(1:psz)] .+= 1

                ip += 1
            end
        end
    end

    # prevent division by 0 error for voxels uncovered by any patch
    Pcount[Pcount .== 0] .= 1

    # Divide each voxel by their number of contributing patches
    img ./= Pcount

    return img
end

"""
patch_nucnorm(P::AbstractArray)

compute the sum of nuclear norm of patches

Inputs:
P: 3D tensor of (space x time) patch matrices. size = (prod(patch_size), Nt, Np)

Outputs:
cost: scalar nuclear norm penalty.
"""
function patch_nucnorm(P::AbstractArray)
    costs = zeros(size(P, ndims(P))) # ur welcome

    @threads for ip = 1:size(P, ndims(P))
        costs[ip] = sum(svdvals(P[:,:,ip]))
    end

    return sum(costs)
end;

"""
SVST(X::AbstractArray, β)

Singular Value Soft-Thresholding
proximal operator to nuclear norm

Inputs:
X: matrix to be low-rankified via SVST

Outputs:
low-rankified version of X
"""
function SVST(X::AbstractMatrix, β)
    U,s,V = svd(X)
    sthresh = @. max(s - β, 0)
    keep = findall(>(0), sthresh)
    return U[:,keep] * Diagonal(sthresh[keep]) * V[:,keep]'
end;

"""
patchSVST(X::AbstractArray, β, patch_size, step_size; prob=1)

apply SVST to in a patch-wise manner to an image
average of proximal operators to the nuclear norm

Inputs:
img: 3D time series data of size (Nx, Ny, Nz, Nt)
β: soft-thresholding threshold hyperparameter
patch_size: length 3 vector describing the side lengths of cubic patches
stride_size: length 3 vector describing the stride lengths between patches

Outputs:
patch-wise low-rankified version of img
"""
function patchSVST(img::AbstractArray, β, patch_size, stride_size; prob=1)
    # extract patches
    P = img2patches(img, patch_size, stride_size)
    
    # low-rankify each patch with probability p
    ips = 1:size(P,5)
    ips = ips[rand(size(P,5)) .<= prob]
    for ip = ips
        P[:,:,ip] = SVST(P[:,:,ip], β) # can this be done in-place for speed?
    end

    # recombine patches into image and return
    return patches2img(P, patch_size, stride_size, size(img)[1:3])
end;