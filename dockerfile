FROM python:3.11.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    mlflow==2.15.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.1 \
    lightgbm==4.5.0 \
    matplotlib==3.9.1 \
    cffi==1.16.0 \
    psutil==6.0.0 \
    pyarrow==15.0.2 \
    scipy==1.14.0 \
    gunicorn==20.1.0

# Copy application files
COPY . .

# Environment variables
ENV FLASK_APP=flask_app/app.py
ENV FLASK_ENV=production
ENV PORT=5000

# Expose port
EXPOSE $PORT

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "flask_app.app:app"]