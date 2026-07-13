# --- stage 1: compile the native reflectivity CMAX kernel --------------------
FROM rust:1-slim AS native
WORKDIR /build
COPY native/cmax.rs .
# Generic x86-64 build (no target-cpu=native) so it can't hit illegal
# instructions on the deploy host.
RUN rustc -C opt-level=3 --crate-type=cdylib cmax.rs -o libcmax.so

# --- stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

# Deps first for layer caching. All pure-wheel installs (no build toolchain).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY radar ./radar
COPY --from=native /build/libcmax.so ./radar/libcmax.so

EXPOSE 8099
CMD ["python", "-m", "uvicorn", "radar.api:app", "--host", "0.0.0.0", "--port", "8099"]
