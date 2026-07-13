FROM python:3.12-slim

WORKDIR /app

# Deps first for layer caching. All pure-wheel installs (no build toolchain).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY radar ./radar

EXPOSE 8099
CMD ["python", "-m", "uvicorn", "radar.api:app", "--host", "0.0.0.0", "--port", "8099"]
