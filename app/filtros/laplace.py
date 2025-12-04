import time
import numpy as np
import pyvips
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def generate_laplacian_kernel(size: int) -> np.ndarray:
    kernel = np.full((size, size), -1, dtype=np.float32)
    center = size // 2
    kernel[center, center] = size * size - 1
    return kernel


def laplace(image: pyvips.Image, ksize: int, bloques: int, threads: int):

    cuda_code = """
    __global__ void laplace_manual(unsigned char* input, unsigned char* output, float *kernel, int width, int height, int ksize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int half = ksize / 2;

        if (x >= width || y >= height) return;

        float sum = 0.0f;
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                float kval = kernel[(ky + half) * ksize + (kx + half)];
                sum += input[iy * width + ix] * kval;
            }
        }
        output[y * width + x] = sum;
    }
    """

    mod = SourceModule(cuda_code)
    laplace_manual = mod.get_function("laplace_manual")

    # pyvips → numpy
    array = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width),
    )
    height, width = array.shape

    # Crear kernel y reservar memoria en GPU
    kernel = generate_laplacian_kernel(ksize)

    output = np.zeros_like(array)

    block_dim = (threads, threads, 1)
    grid_dim = (
        (width + block_dim[0] - 1) // block_dim[0],
        (height + block_dim[1] - 1) // block_dim[1],
    )

    # Ejecutar kernel y medir tiempo
    start = time.time()
    laplace_manual(
        cuda.In(array),
        cuda.Out(output),
        cuda.In(kernel),
        np.int32(width),
        np.int32(height),
        np.int32(ksize),
        block=block_dim,
        grid=grid_dim,
    )
    cuda.Context.synchronize()
    elapsed = time.time() - start

    # Copiar resultado de GPU → CPU
    result_array = np.empty_like(output)

    # Normalizar y convertir a 8-bit
    result_array = np.clip(output, 0, 255).astype(np.uint8)

    # numpy → pyvips
    result_image = pyvips.Image.new_from_memory(
        result_array.tobytes(), width, height, 1, "uchar"
    )

    return result_image, elapsed
