import time
import numpy as np
import pyvips
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


def createGaussianKernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size//2
    sum = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            value = np.exp(-(x*x + y*y) / (2.0 * sigma**2))
            kernel[i, j] = value
            sum += value

    kernel /= sum
    return kernel


def gaussiano(image: pyvips.Image, ksize: int, sigma: int, bloques: int, threads: int):
    mod = SourceModule("""
        __global__ void gaussianoParalelo(unsigned char* input, unsigned char* output, float* kernel, int width, int height, int kSize) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if(x >= width || y >= height) return;
            int centro = kSize / 2;
            float sum = 0.0f;

            for(int ky = -centro; ky <= centro; ky++) {
                for(int kx = -centro; kx <= centro; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if(px >= 0 && px < width && py >= 0 && py < height) {
                        float valor = kernel[(ky + centro) * kSize + (kx + centro)];
                        sum += valor * input[py * width + px];
                    }
                }
            }
            sum = fminf(fmaxf(sum, 0.0f), 255.0f);
            output[y * width + x] = (unsigned char) sum;
        }
    """)



    array = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width)
    )
    kernel = createGaussianKernel(ksize, sigma)

    result = np.zeros_like(array)
    block = (threads, threads, 1)
    grid = (
        (array.shape[1] + block[0] - 1) // block[0],
        (array.shape[0] + block[1] - 1) // block[1],
        1
    )
    paralelo = mod.get_function("gaussianoParalelo")
    inicio = time.time()
    paralelo(drv.In(array), drv.Out(result), drv.In(kernel), np.int32(array.shape[1]), np.int32(array.shape[0]), np.int32(ksize), block=block, grid=grid)
    final = time.time()
    total = final - inicio

    output = pyvips.Image.new_from_memory(
        result.tobytes(), image.width, image.height, 1, "uchar"
    )

    return output, total

