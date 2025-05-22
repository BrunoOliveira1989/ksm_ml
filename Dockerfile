# Imagem base minimalista do Python
FROM python:3.11-slim

# Evita cache de bytecode e buffer de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências de sistema necessárias para scikit-learn e psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libpq-dev \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala as dependências Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expor a porta que o app irá rodar
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

