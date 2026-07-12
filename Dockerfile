# Start from a slim, official Python base image.
# "slim" = smaller image size than the full python image, still has what we need.
FROM python:3.12-slim

# Set the working directory inside the container.
# All following commands run relative to this path, like "cd" for the container.
WORKDIR /app

# Copy just requirements.txt first (not the whole project yet).
# Why: Docker caches each step. If only your code changes (not dependencies),
# this lets Docker skip re-installing packages on every rebuild - much faster.
COPY requirements.txt .

# Install PyTorch's CPU-only build explicitly, from PyTorch's own package
# index. Without this, pip defaults to a CUDA/GPU-enabled torch build on
# Linux (which is what runs inside this container) - multiple GB in size
# and unnecessary since we're not using a GPU. Installing this first means
# sentence-transformers (which depends on torch) will find this version
# already satisfied and won't try to pull in the much larger CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project code into the container.
COPY src/ ./src/

# Streamlit's default port. We tell Docker this container listens on 8501.
EXPOSE 8501

# The command that runs when the container starts.
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]