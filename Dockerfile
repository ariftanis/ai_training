# Use the official NVIDIA CUDA base image that matches the system environment
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python (generic), the venv module, and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    cmake \
    libcurl4-openssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Create a virtual environment and add it to the PATH
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies into the virtual environment
COPY requirements.txt .
# The 'pip' command will now automatically use the one from the virtual environment
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# Copy the source code and the entrypoint script
COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint to our startup script
ENTRYPOINT ["./entrypoint.sh"]

# Set the default command to run the inference script.
# This command will be executed by the entrypoint script after the training check.
CMD ["python3", "src/inference.py", "Sancaktepe belediye başkanı kimdir?"]