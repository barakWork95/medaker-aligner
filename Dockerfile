FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1 \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .

# Memory settings (see README → Memory): one torch thread, tight glibc arenas, no OMP fan-out.
ENV TORCH_HOME=/srv/.torch \
    TORCH_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MMS_FA_PREFER_INT8=1 \
    PORT=8000

EXPOSE 8000
CMD ["python", "main.py"]