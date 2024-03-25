FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY combiner_whisper.py .

CMD ["python", "combiner_whisper.py"]
