# Imagen base con Python y libGL ya incluido
FROM python:3.12-slim

# Instalar dependencias del sistema necesarias para opencv y demás
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear el directorio de la app
WORKDIR /app

# Copiar los archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar el bot
CMD ["python", "bot.py"]
