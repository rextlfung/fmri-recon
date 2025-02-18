#=
Collection of functions to be used on iterative image reconstruction.
=#

# patch extractor: L part -> patches of L part
# using cubic patches for simplicity
function img2patches(img::AbstractArray, patch_size, step_size)
    # Unpack inputs
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = step_size

    # calculate number of steps in each direction
    Nsteps_x = fld(Nx - psx, ssx)
    Nsteps_y = fld(Ny - psy, ssy)
    Nsteps_z = fld(Nz - psz, ssz)

    # calculate number of patches to extract
    npatches = (nsteps_x + 1)*(nsteps_y + 1)*(nsteps_z + 1)

    # preallocate output array to hold extracted patches
    P = zeros(ComplexF32, (psx, psy, psz, nt, npatches))

    # slide through L and extract patches
    patch_num = 1 # patch counter
    for iz in 0:nsteps_z
        for iy in 0:nsteps_y
            for ix in 0:nsteps_x
                P[:,:,:,:,patch_num] = L[ix*ssx .+ (1:psx), iy*ssy .+ (1:psy), iz*ssz .+ (1:psz), :]
                patch_num += 1
            end
        end
    end

    return P
end

