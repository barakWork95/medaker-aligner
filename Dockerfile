FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1 \
 && pip install --no-cache-dir -r requirements.txt

# Bake the MMS_FA weights (~1.2 GB) into the image so the container starts instantly.
# Build with --build-arg PRELOAD=0 to skip and download on first start instead.
ARG PRELOAD=1
ENV TORCH_HOME=/srv/.torch
RUN if [ "$PRELOAD" = "1" ]; then python -c "import torchaudio; torchaudio.pipelines.MMS_FA.get_model(with_star=False)"; fi

COPY app ./app
COPY main.py .
EXPOSE 8000
ENV PORT=8000
CMD ["python", "main.py"]
