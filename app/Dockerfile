FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Sistema base
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential libvips && \
    apt-get clean

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Crear directorio
WORKDIR /code

# Copiar dependencias primero (mejor cache)
COPY requirements.txt /code/requirements.txt

# Instalar dependencias
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# Copiar código
COPY . /code/app

# Puerto FastAPI
EXPOSE 5000

# Comando de ejecución
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

