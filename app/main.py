from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.image_router import router as image_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(image_router)

@app.get("/")
def root():
    return {"message": "Hola Mundo desde FastAPI"}

@app.get("/hola/{nombre}")
def read_item(nombre):
    return {"saludo": f"Hola, {nombre}"}
