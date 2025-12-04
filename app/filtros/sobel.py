import time
import numpy as np
import pyvips
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


def createSobelKernels(size, sigma):
    center = size // 2

    # Derivadas aproximadas gaussianas
    Kx = np.zeros((size, size), dtype=np.float32)
    Ky = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            gx = -x * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gy = -y * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            Kx[i, j] = gx
            Ky[i, j] = gy

    # Normalización para evitar valores demasiado grandes
    Kx /= np.sum(np.abs(Kx))
    Ky /= np.sum(np.abs(Ky))
    return Kx, Ky


def sobel(image: pyvips.Image, ksize: int, sigma: float, bloques: int, threads: int):

    # Convertir a numpy array
    array = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width),
    )

    # Crear kernels Sobel
    Kx, Ky = createSobelKernels(ksize, sigma)

    # Compilar kernel CUDA
    mod = SourceModule(
        """
        __global__ void sobelParalelo(unsigned char* input, unsigned char* output,
                                      float* kernelX, float* kernelY,
                                      int width, int height, int kSize) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            int half = kSize / 2;
            float gx = 0.0f;
            float gy = 0.0f;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        unsigned char val = input[py * width + px];
                        gx += val * kernelX[(ky + half) * kSize + (kx + half)];
                        gy += val * kernelY[(ky + half) * kSize + (kx + half)];
                    }
                }
            }

            float mag = sqrtf(gx * gx + gy * gy);
            mag = fminf(fmaxf(mag, 0.0f), 255.0f);
            output[y * width + x] = (unsigned char)mag;
        }
        """
    )

    # Configurar ejecución CUDA
    result = np.zeros_like(array)
    block = (threads, threads, 1)
    grid = (
        (array.shape[1] + block[0] - 1) // block[0],
        (array.shape[0] + block[1] - 1) // block[1],
        1,
    )

    sobel_gpu = mod.get_function("sobelParalelo")

    # Ejecutar y medir el tiempo
    inicio = time.time()
    sobel_gpu(
        drv.In(array),
        drv.Out(result),
        drv.In(Kx.astype(np.float32)),
        drv.In(Ky.astype(np.float32)),
        np.int32(array.shape[1]),
        np.int32(array.shape[0]),
        np.int32(ksize),
        block=block,
        grid=grid,
    )
    final = time.time()
    total = final - inicio

    # Convertir resultado a pyvips.Image
    output_image = pyvips.Image.new_from_memory(
        result.tobytes(), image.width, image.height, 1, "uchar"
    )

    return output_image, total
