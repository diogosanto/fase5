FROM python:3.10-slim

WORKDIR /app

# Copia apenas arquivos essenciais primeiro (melhor cache)
COPY pyproject.toml ./
COPY api ./api
COPY src ./src

# Instala o projeto e dependências definidas no pyproject.toml
RUN pip install --no-cache-dir .

# Comando para rodar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
