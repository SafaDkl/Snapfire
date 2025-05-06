FROM python:3.13

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git build-essential ninja-build cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy your code
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install opencv-python diffusers[torch] transformers accelerate && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
# Set the command to run your app
CMD ["python", "main.py"]
