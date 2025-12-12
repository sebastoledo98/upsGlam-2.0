import time
import os
import numpy as np
import pyvips
import pycuda.driver as drv
import pycuda.autoinit  # inicializa el contexto CUDA
from pycuda.compiler import SourceModule

# Se asume ups.png en el mismo directorio que este archivo
LOGO_PATH = os.path.join(os.path.dirname(__file__), "ups.png")

KERNEL_LOGO = r"""
extern "C"
__global__ void aplicar_logo(
    unsigned char* input,
    unsigned char* logo,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int logo_w,
    int logo_h,
    float alpha
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    int lx = x % logo_w;
    int ly = y % logo_h;
    int lidx = ly * logo_w + lx;

    unsigned char logo_px = logo[lidx];

    for (int c = 0; c < channels; ++c) {
        float orig = (float)input[idx + c];
        float out = (1.0f - alpha) * orig + alpha * (float)logo_px;
        if (out < 0.0f) out = 0.0f;
        if (out > 255.0f) out = 255.0f;
        output[idx + c] = (unsigned char)(out);
    }
}
""";

_module_logo = SourceModule(KERNEL_LOGO)
_kernel_logo = _module_logo.get_function("aplicar_logo")


def _load_logo_gray(logo_size: int = 96) -> np.ndarray:
    """
    Carga el logo en escala de grises y lo reescala
    a un tamaño pequeño para usar como marca de agua repetida.
    """
    if not os.path.exists(LOGO_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de logo en: {LOGO_PATH}")

    logo = pyvips.Image.new_from_file(LOGO_PATH, access="sequential")
    logo = logo.colourspace("b-w")

    max_dim = max(logo.width, logo.height)
    if max_dim == 0:
        raise ValueError("El logo tiene dimensiones inválidas.")

    scale = float(logo_size) / float(max_dim)
    if scale <= 0.0:
        scale = 1.0

    logo_resized = logo.resize(scale)

    logo_array = np.ndarray(
        buffer=logo_resized.write_to_memory(),
        dtype=np.uint8,
        shape=(logo_resized.height, logo_resized.width),
    )

    return logo_array


def logo(
    image: pyvips.Image,
    ksize: int,
    sigma: float,
    bloques: int,
    threads: int,
    logo_size: int = 96,
    alpha: float = 0.85,
):
    """
    Aplica el logo como marca de agua repetida sobre la imagen.
    La imagen conserva sus colores originales y el logo se usa en escala de grises.
    """
    # Normalizar canales
    if image.bands == 1:
        image = image.colourspace("b-w")
        image = image.bandjoin([image, image])
    elif image.bands == 4:
        image = image.flatten()

    if image.bands != 3:
        image = image.colourspace("srgb")

    width = image.width
    height = image.height
    channels = image.bands

    # Imagen a numpy (H, W, C)
    array = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width, channels),
    )

    # Logo gris
    logo_array = _load_logo_gray(logo_size=logo_size)
    logo_h, logo_w = logo_array.shape

    input_flat = array.astype(np.uint8).ravel()
    logo_flat = logo_array.astype(np.uint8).ravel()
    output_flat = np.zeros_like(input_flat)

    block = (threads, threads, 1)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        1,
    )

    start = time.time()
    _kernel_logo(
        drv.In(input_flat),
        drv.In(logo_flat),
        drv.Out(output_flat),
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        np.int32(logo_w),
        np.int32(logo_h),
        np.float32(alpha),
        block=block,
        grid=grid,
    )
    end = time.time()
    total = end - start

    result_array = output_flat.reshape((height, width, channels))

    output = pyvips.Image.new_from_memory(
        result_array.tobytes(), width, height, channels, "uchar"
    )

    return output, total
