FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY fastapi_service.py .
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]