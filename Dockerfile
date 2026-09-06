# AI 이미지 (TASK-12)
FROM python:3.11-slim

WORKDIR /app

# opencv(libgl1/libglib2.0-0), moviepy/yt-dlp(ffmpeg), pytesseract(tesseract-ocr),
# deno 설치(curl/unzip)가 필요로 하는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp EJS(YouTube JS 챌린지 솔버)가 요구하는 JS 런타임 (TASK-16)
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh
ENV PATH="/usr/local/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
