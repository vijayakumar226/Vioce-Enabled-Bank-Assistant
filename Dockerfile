FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy full project
COPY . /app

# Install backend dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Install frontend dependencies and build
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Copy nginx config
WORKDIR /app
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
