FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy API application
COPY api/ ./api/

# Copy trained BigMart model
COPY models/ ./models/

# Create logs directory
RUN mkdir -p /app/logs

# Set model path
ENV BIGMART_MODEL_PATH=/app/models/bigmart_model.pkl

# Expose API port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]