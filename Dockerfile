FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p database/chroma_store

# Make startup script executable
RUN chmod +x docker/startup.sh

EXPOSE 5001

ENV PYTHONPATH=/app
ENV FLASK_APP=server/app.py
ENV FLASK_ENV=production

CMD ["sh", "-c", "python server/app.py & while ! curl -f http://localhost:5001/collections; do sleep 1; done && cd database && python initialize.py && wait"] 
