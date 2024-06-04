# Example: Backend Dockerfile

# Python 3.10 Slim Bullseye sürümünü baz al
FROM python:3.10-slim-bullseye

# Sistem bağımlılıklarını yükle
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        gcc \
        python3-dev \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Pip'i güncelle
RUN pip install --no-cache-dir --upgrade pip

# Çalışma dizinini belirle
WORKDIR /app

# requirements.txt dosyasını kopyala ve bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Uygulamanın çalışacağı portu belirle (gerekirse)
EXPOSE 5003

# Uygulamayı başlat
CMD ["python", "app.py"]
