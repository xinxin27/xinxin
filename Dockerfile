FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FC_SERVER_PORT=9000

# —— 强制使用国内镜像（阿里云），并禁用 deb822 默认源 —— 
RUN set -eux; \
    rm -f /etc/apt/sources.list.d/debian.sources || true; \
    printf '%s\n' \
      'deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware' \
      'deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware' \
      'deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware' \
      > /etc/apt/sources.list; \
    printf 'Acquire::Retries "10";\nAcquire::http::No-Cache "true";\nAcquire::https::No-Cache "true";\n' \
      > /etc/apt/apt.conf.d/99retries; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
    libgl1 libglib2.0-0; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

# 先装 CPU 版 torch/torchvision，再装其余依赖
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r requirements.txt

# Copy application files
COPY app.py .
COPY bootstrap .
COPY bird_detection ./bird_detection
COPY templates ./templates
COPY static ./static
COPY secret.env .

# Handle Windows line endings
RUN sed -i 's/\r$//' bootstrap

EXPOSE 9000
CMD ["/code/bootstrap"]
