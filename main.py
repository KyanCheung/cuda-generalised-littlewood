from numba import *
from numba import cuda
import numpy as np
from cmath import *
import time
import matplotlib.pyplot as plt

# Information about the GPU that this is being run on, to help optimise things.
SM = 48
wrapPerSM = 48

n = 24
arrayWidth = n + (not n % 2) # Ensure array width is always odd to reduce bank collisions

a = -1

numPoly = 2 ** n
# Since any found real roots are approximated, it is best to use an odd number
pixels = 1001


# This represents how many polynomials are worked on simultaneously by a block.
# WARNING: SHOULD BE A POWER OF 2 OR ELSE THE GRID STRIDE METHOD WOULD FAIL
chunks = 32

# This generates the starting values for z
startz = np.array([np.exp(1j * (1 + tau * k / n)) for k in range(n)], dtype=np.csingle)

@cuda.jit
def roots(d_z, c):
    # Saving polynomials 
    s_p = cuda.shared.array(shape=(chunks, arrayWidth), dtype=float32)
    # Saving polynomial derivative
    s_pdiff = cuda.shared.array(shape=(chunks, arrayWidth), dtype=complex64)
    # Saving estimations of roots
    s_z = cuda.shared.array(shape=(chunks, arrayWidth), dtype=complex64)
    # This tells us how many roots are not close enough - when threads=0 we can proceed
    s_threads = cuda.shared.array(shape=1, dtype=u8)

    c_startz = cuda.const.array_like(startz)

    x, y = cuda.grid(2)
    # This will represent which polynomial we are working on
    tx = cuda.threadIdx.x
    # This will represent the root we are working on
    ty = cuda.threadIdx.y
    # This is an "offset" based on the index of the block
    bx = chunks * cuda.blockIdx.x

    # We are using a grid-stride method to recycle threads
    for offset in range(bx, numPoly, chunks*cuda.gridDim.x):
        # Generating s_p on the fly
        s_p[tx][ty] = 1 if ((tx+offset) >> ty) & 1 else c
        # Wait until s_p is fully initialised by each thread
        cuda.syncthreads()

        if ty == n-1:
           s_pdiff[tx][ty] = n
        else:
            s_pdiff[tx][ty] = (ty + 1) * s_p[tx][ty+1]
        # Wait until s_p is fully initialised by each thread
        cuda.syncthreads()

        # Start with estimates of z where |z|=1 but not roots of unity
        s_z[tx][ty] = c_startz[ty]
        cuda.syncthreads()

        # Setting up variables to evaluate p and p' via Horner's method
        pz = 1
        pdiffz = 0
        sz = s_z[tx][ty]
        for i in range(n-1, -1, -1):
            pz = s_p[tx][i] + pz * sz
            pdiffz = s_pdiff[tx][i] + pdiffz * sz

        # Evaluate p(z)/p'(z) at each z
        # Note: p(z) will have at most relative error of 4n^2*mu+O(n*mu) where mu=2^-23 (for single precision). 
        # Shouldn't be too significant at n<30 but double precision would be nice.
        ratio = pz / pdiffz
        s_threads[0] = chunks * n
        cuda.syncthreads()

        # Keep iterating until all roots are decently close
        # By Bini (1996) |np(z)/p'(z)| approximates how far z is from a root. 
        # Surely this is a more widespread result but I can't find.
        stop = False # Flips to True when condition met
        while s_threads[0] > 0:
            if not stop:
                sumInvRoot = 0
                for k in range(n):
                    if ty != k:
                        sumInvRoot += 1/(sz - s_z[tx][k])
                w = ratio / (1 - (ratio * sumInvRoot))
            # We want to make sure that we do not change s_z while other threads are working
            cuda.syncthreads()
            if not stop:
                s_z[tx][ty] -= w
                sz -= w

                # To improve numerical stability, for |z|>1 we use an alternate method of calculating ratio
                if sz.real**2 + sz.imag**2 <= 1:
                    pz = 1
                    pdiffz = 0
                    for i in range(n-1, -1, -1):
                        pz = s_p[tx][i] + pz * sz
                        pdiffz = s_pdiff[tx][i] + pdiffz * sz
                else:
                    pz = s_p[tx][0]
                    pdiffz = 0
                    y = 1 / sz
                    for i in range(n-1):
                        pz = s_p[tx][i+1] + pz * y
                        pdiffz = s_pdiff[tx][i] + pdiffz * y
                    pz += sz
                    pdiffz = s_pdiff[tx][n-1] + pdiffz * y

                ratio = pz / pdiffz
                if n * abs(ratio) < 2 ** -18:
                    stop = True
                    # Make sure writes are not overwriting each other
                    cuda.atomic.add(s_threads, 0, -1)
            cuda.syncthreads()

        # We want to discard roots that are not in the grid
        x = s_z[tx][ty].real
        y = s_z[tx][ty].imag
        if -2 <= x <= 2 and -2 <= y <= 2:
            # Position of found root on image
            xpos = int(pixels * (x + 2) / 4)
            ypos = int(pixels * (y + 2) / 4)
            cuda.atomic.add(d_z, (ypos, xpos), 1)

blocks = int(32 * wrapPerSM / (n * chunks)) * SM
blocksPerGrid = (blocks, 1)
threadsPerBlock = (chunks, n)

d_z = cuda.device_array(shape=(pixels, pixels), dtype=np.uint)

times = [time.perf_counter()]
roots[blocksPerGrid, threadsPerBlock](d_z, a)
z = d_z.copy_to_host()

np.save(f"littlewood{n}_{a}.npy", z)
print(f"Time taken: {time.perf_counter()-times[0]}s")

im = plt.imsave(f"littlewood{n}_{1}.png", np.log(z+1), vmax=2, cmap="hot")

print("Completely done!")