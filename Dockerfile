FROM python:3.11-slim

WORKDIR /app

# These system libs are needed for OpenCV to work properly in a headless environment
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copies requirements first so Docker can cache the pip install step separately
COPY requirements.txt .

# Installs all the Python packages the app depends on
RUN pip install --no-cache-dir -r requirements.txt

# Brings in the actual application code after dependencies are installed
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]