FROM python:3.11-slim

# Keep container logs immediate and avoid writing .pyc files into mounted folders.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies before copying the full project so Docker can reuse this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, data, and local scripts after dependencies are installed.
COPY . .

# FastAPI listens on this port by default when the image runs without overrides.
EXPOSE 8000

# The image defaults to the API; docker-compose overrides this for the dashboard.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
