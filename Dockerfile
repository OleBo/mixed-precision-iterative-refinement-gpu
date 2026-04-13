# Multi-stage build: compile CUDA code
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy project source
COPY . .

# Create build directory and compile
RUN rm -rf build && mkdir -p build && cd build && \
    cmake .. && \
    make

# Final runtime image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Install runtime dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install numpy matplotlib

# Set working directory
WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder /build/build /app/build
COPY --from=builder /build/include /app/include
COPY --from=builder /build/python /app/python
COPY --from=builder /build/results /app/results
COPY --from=builder /build/CMakeLists.txt /app/

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "cd /app && python3 python/experiments.py"]
