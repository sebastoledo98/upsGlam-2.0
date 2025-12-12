import time
import numpy as np
import pyvips
import pycuda.driver as drv
import pycuda.autoinit  # inicializa el contexto CUDA
from pycuda.compiler import SourceModule

KERNEL_CARIC = r"""
extern "C"
__global__ void edges_pencil(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    int gx = 0;
    int gy = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = x + kx;
            int yy = y + ky;

            if (xx < 0) xx = 0;
            if (xx >= width) xx = width - 1;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;

            int k_idx = yy * width + xx;
            unsigned char p = input[k_idx];

            int sx = 0;
            int sy = 0;

            if (ky == -1 && kx == -1) { sx = -1; sy = -1; }
            else if (ky == -1 && kx == 0) { sx = 0; sy = -2; }
            else if (ky == -1 && kx == 1) { sx = 1; sy = -1; }
            else if (ky == 0 && kx == -1) { sx = -2; sy = 0; }
            else if (ky == 0 && kx == 0) { sx = 0; sy = 0; }
            else if (ky == 0 && kx == 1) { sx = 2; sy = 0; }
            else if (ky == 1 && kx == -1) { sx = -1; sy = 1; }
            else if (ky == 1 && kx == 0) { sx = 0; sy = 2; }
            else if (ky == 1 && kx == 1) { sx = 1; sy = 1; }

            gx += sx * (int)p;
            gy += sy * (int)p;
        }
    }

    float mag = sqrtf((float)(gx * gx + gy * gy));

    float edge = mag * 0.5f;
    if (edge > 255.0f) edge = 255.0f;

    float pencil = 255.0f - edge;

    if (pencil < 0.0f) pencil = 0.0f;
    if (pencil > 255.0f) pencil = 255.0f;

    output[idx] = (unsigned char)(pencil);
}
""";

_module_caric = SourceModule(KERNEL_CARIC)
_kernel_caric = _module_caric.get_function("edges_pencil")


def caricatura(
    image: pyvips.Image,
    ksize: int,
    sigma: float,
    bloques: int,
    threads: int,
):
    """
    Efecto de dibujo a lápiz en escala de grises usando Sobel en GPU.
    """
    if image.bands != 1:
        image = image.colourspace("b-w")

    height = image.height
    width = image.width

    array = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width),
    )

    result = np.zeros_like(array)

    block = (threads, threads, 1)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        1,
    )

    start = time.time()
    _kernel_caric(
        drv.In(array),
        drv.Out(result),
        np.int32(width),
        np.int32(height),
        block=block,
        grid=grid,
    )
    end = time.time()
    total = end - start

    output = pyvips.Image.new_from_memory(
        result.tobytes(), width, height, 1, "uchar"
    )

    return output, total
