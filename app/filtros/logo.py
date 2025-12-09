import os
import time
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


LOGO_FILENAME = "ups.png"
LOGO_PATH = os.path.join(os.path.dirname(__file__), LOGO_FILENAME)


KERNEL = r"""
__global__ void blend_watermark_rgb(
    const unsigned char *orig,
    const unsigned char *logo,
    unsigned char *out,
    int width,
    int height,
    float alpha
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * 3;

    // Mezclar por canal (RGB)
    for (int c = 0; c < 3; c++) {
        float o = (float)orig[idx + c];
        float l = (float)logo[idx + c];

        float val = alpha * o + (1.0f - alpha) * l;

        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;

        out[idx + c] = (unsigned char)val;
    }
}
""";


_module = SourceModule(KERNEL)
_blend_kernel = _module.get_function("blend_watermark_rgb")


def make_tiled_logo(width, height, logo_size=100):
    """
    Genera un mosaico con el logo repetido en escala de grises
    sobre un fondo blanco, del mismo tamaño que la imagen original.
    """
    if not os.path.exists(LOGO_PATH):
        raise FileNotFoundError(f"No se encontró el logo: {LOGO_PATH}")

    # Abrimos el logo y lo pasamos a ESCALA DE GRISES
    logo = Image.open(LOGO_PATH).convert("L")  # gris
    logo = logo.resize((logo_size, logo_size))

    # Lo convertimos a RGB para poder mezclarlo con la imagen a color
    logo_rgb = logo.convert("RGB")

    # Fondo blanco RGB
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    # Repetimos el logo en mosaico
    for y in range(0, height, logo_size):
        for x in range(0, width, logo_size):
            canvas.paste(logo_rgb, (x, y))

    return canvas



def watermark_tiled(
    input_path: str,
    output_path: str,
    logo_size: int = 100,  # tamaño de cada logo
    alpha: float = 0.85,   # 0.85 = imagen original fuerte, watermark suave
    threads: int = 16,
):
    """
    Marca de agua repetida (tiled watermark)
    usando Pillow + PyCUDA en RGB.
    """

    # Leer imagen original (color)
    img = Image.open(input_path).convert("RGB")
    orig = np.array(img, dtype=np.uint8)
    height, width, _ = orig.shape

    # Generar mosaico repetido del logo
    tiled = make_tiled_logo(width, height, logo_size)
    tiled_np = np.array(tiled, dtype=np.uint8)

    # Reservar memoria en GPU
    d_orig = cuda.mem_alloc(orig.nbytes)
    d_logo = cuda.mem_alloc(tiled_np.nbytes)
    d_out = cuda.mem_alloc(orig.nbytes)

    cuda.memcpy_htod(d_orig, orig)
    cuda.memcpy_htod(d_logo, tiled_np)

    block = (threads, threads, 1)
    grid = (
        (width + threads - 1) // threads,
        (height + threads - 1) // threads,
        1,
    )

    alpha32 = np.float32(alpha)

    start = time.time()
    _blend_kernel(
        d_orig,
        d_logo,
        d_out,
        np.int32(width),
        np.int32(height),
        alpha32,
        block=block,
        grid=grid,
    )
    cuda.Context.synchronize()
    elapsed = time.time() - start

    
    out = np.empty_like(orig)
    cuda.memcpy_dtoh(out, d_out)

    # Guardar imagen final
    Image.fromarray(out).save(output_path)

    return elapsed
