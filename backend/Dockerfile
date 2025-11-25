FROM python:3.11-slim
WORKDIR /app
COPY ./app /app
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn[standard] python-multipart pillow torch torchvision
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
