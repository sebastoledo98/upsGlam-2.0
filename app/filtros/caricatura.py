import time
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


KERNEL = r"""
__global__ void caricature_filter(
    unsigned char *input,
    unsigned char *output,
    int width,
    int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evitar acceder fuera de la imagen
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1)
    {
        if (x < width && y < height)
        {
            int idx_b = y * width + x;
            output[idx_b] = 255; // borde de la imagen en blanco
        }
        return;
    }

    int idx = y * width + x;

    // 1) Gradiente Sobel en x
    int gx =
        -1 * input[(y-1)*width + (x-1)] + 1 * input[(y-1)*width + (x+1)]
      + -2 * input[(y)*width   + (x-1)] + 2 * input[(y)*width   + (x+1)]
      + -1 * input[(y+1)*width + (x-1)] + 1 * input[(y+1)*width + (x+1)];

    // 2) Gradiente Sobel en y
    int gy =
        -1 * input[(y-1)*width + (x-1)] + -2 * input[(y-1)*width + (x)] + -1 * input[(y-1)*width + (x+1)]
      +  1 * input[(y+1)*width + (x-1)] +  2 * input[(y+1)*width + (x)] +  1 * input[(y+1)*width + (x+1)];

     // 3) Magnitud aproximada del gradiente
    float mag = fabsf((float)gx) + fabsf((float)gy);

    // Escalamos la magnitud a rango [0, 255]
    mag = mag / 1024.0f * 255.0f;

    // Aumentar contraste de bordes (factor > 1.0 hace bordes mas fuertes)
    mag = mag * 1.8f;
    if (mag > 255.0f)
        mag = 255.0f;

    // 4) Invertir: bordes mas oscuros sobre fondo claro (efecto lapiz)
    unsigned char edge = (unsigned char)(255.0f - mag);

    output[idx] = edge;
}
""";



_module = SourceModule(KERNEL)
_caricature = _module.get_function("caricature_filter")


def caricatura(image_path, output_path, threads=16):
    """
    Filtro de caricatura en escala de grises usando PyCUDA.
    1) Sobel en GPU para detectar bordes.
    2) Posterizacion de intensidades.
    3) Bordes negros + zonas internas con tonos planos.
    """
    start = time.time()

    # Leer imagen y convertir a escala de grises
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    height, width = arr.shape

    # Reservar memoria en GPU
    d_input = cuda.mem_alloc(arr.nbytes)
    d_output = cuda.mem_alloc(arr.nbytes)

    cuda.memcpy_htod(d_input, arr)

    # Configuracion de ejecucion
    block = (threads, threads, 1)
    grid = ((width + threads - 1) // threads,
            (height + threads - 1) // threads,
            1)

    # Ejecutar kernel
    _caricature(
        d_input,
        d_output,
        np.int32(width),
        np.int32(height),
        block=block,
        grid=grid
    )

    cuda.Context.synchronize()

    # Copiar resultado de vuelta a CPU
    result = np.empty_like(arr)
    cuda.memcpy_dtoh(result, d_output)

    # Guardar imagen
    Image.fromarray(result).save(output_path)

    elapsed = time.time() - start
    return elapsed
