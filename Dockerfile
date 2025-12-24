# Use the official NVIDIA CUDA base image that matches the system environment
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python and other system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# Copy the source code and the entrypoint script
COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint to our startup script
ENTRYPOINT ["./entrypoint.sh"]

# Set the default command to run the inference script.
# This command will be executed by the entrypoint script after the training check.
CMD ["python3", "src/inference.py", "Sancaktepe belediye başkanı kimdir?"]