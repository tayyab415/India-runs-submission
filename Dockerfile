FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-sandbox.txt ./
RUN pip install --no-cache-dir -r requirements-sandbox.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
