from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.filtros.gauss import gaussiano
from app.filtros.laplace import laplace
from app.filtros.sobel import sobel
from app.filtros.sharpen import sharpen
import pyvips
import base64

router = APIRouter(prefix="/filtros", tags=["filtros"])

FILTROS = [
    "gaussiano",
    "laplace",
    "sharpen",
    "sobel"
]

@router.post("/procesar_imagen" )
async def procesar_imagen(
        file: UploadFile = File(...),
        filtro: str = Form(...),
        mask_size: int = Form(...),
        sigma: float = Form(...),
        blocks: int = Form(...),
        threads: int = Form(...)
):
    image_bytes = await file.read()

    try:
        image = pyvips.Image.new_from_buffer(image_bytes, "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer imagen: {e}")

    image_gray = image.colourspace("b-w")

    if filtro == "gaussiano":
        output, tiempo = gaussiano(image_gray, mask_size, sigma, blocks, threads)
    elif filtro == "sobel":
        output, tiempo = sobel(image_gray, mask_size, sigma, blocks, threads)
    elif filtro == "laplace":
        output, tiempo = laplace(image_gray, mask_size, blocks, threads)
    elif filtro == "sharpen":
        output, tiempo = sharpen(image_gray, mask_size, blocks, threads)

    buffer_gray = image_gray.jpegsave_buffer(Q=90)
    gray_base64 = base64.b64encode(buffer_gray).decode("utf-8")

    buffer_filtro = output.jpegsave_buffer(Q=90)
    filtro_base64 = base64.b64encode(buffer_filtro).decode("utf-8")

    respuesta = {
        "imagen_original": gray_base64,
        "imagen_filtro": filtro_base64,
        "filtro": filtro,
        "tiempo": tiempo,
        "mask": mask_size,
        "sigma": sigma,
        "blocks": blocks,
        "threads": threads
    }

    return JSONResponse(content=respuesta)
