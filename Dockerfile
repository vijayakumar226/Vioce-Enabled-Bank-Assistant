FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    nodejs \
    npm \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy full project
COPY . /app

# Install backend dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Download embedding model during build so it is cached
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Build FAISS index from documents during Docker build
RUN python apps/build_index.py

# Install frontend dependencies and build
WORKDIR /app/frontend
RUN npm install
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# Copy nginx config
WORKDIR /app
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860
CMD ["/app/start.sh"]