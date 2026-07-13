# --- stage 1: compile the native kernels (generic x86-64, no target-cpu) -----
FROM rust:1-slim AS native
WORKDIR /build
COPY native/ ./native
RUN cd native && cargo build --release

# --- stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

# Deps first for layer caching. All pure-wheel installs (no build toolchain).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY radar ./radar
COPY --from=native /build/native/target/release/libradar_native.so ./radar/libradar_native.so

EXPOSE 8099
CMD ["python", "-m", "uvicorn", "radar.api:app", "--host", "0.0.0.0", "--port", "8099"]
