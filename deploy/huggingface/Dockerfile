FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DEMO_MODE=synthetic
ENV PORT=7860
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 7860
CMD ["python", "deploy/huggingface/app.py"]
